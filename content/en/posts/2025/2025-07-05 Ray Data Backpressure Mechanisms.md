+++
date = '2025-07-05T15:03:26+08:00'
title = 'Ray Data Backpressure Mechanisms'
slug = 'ray-data-backpressure-mechanisms'
translationKey = 'ray-data-backpressure-mechanisms'
author = "sword865"
type = "post"
tags = ["Ray", "Data"]
topics = ["Distributed Computing"]
+++

I have been working on Ray Platform for almost two years now and have run into all sorts of issues. I want to write down some of the common pitfalls, starting with Ray Data.

The most common OOM and OOD problems in Ray Data are usually related to backpressure. In fact, the backpressure story here is not just one mechanism. It has several layers:

1. **Ray Core Generator**: controls Ray Generators so that too much data is not produced in the background, which would otherwise cause OOM or OOD.
2. **Streaming Executor + Resource Allocator**:
   * controls the output rate of running tasks, so one task does not produce too much data at once
   * controls how many tasks a single operator may submit when resources are tight
3. **Backpressure Policies**: additional task-submission rules on top of the core resource checks.

Let us go through them one by one.

# Ray Core Generator: Backpressure on Object Count

[Ray Generator](https://docs.ray.io/en/latest/ray-core/ray-generator.html) is similar to a Python generator in that it can be iterated over, but there is one major difference: Ray Generator uses `ObjectRefGenerator` and continues to run in the background.

That means if one Ray Data `read_task` is reading a large file, we cannot control its memory footprint just by slowing down downstream consumption. Even if the downstream consumer stops pulling, the task can keep producing new blocks.

To address this, Ray Generator supports a configurable threshold, `_generator_backpressure_num_objects`, to apply backpressure.

The core logic is in `HandleReportGeneratorItemReturns` inside `task_manager.cc`. The function is fairly involved because it also handles ordering and idempotency issues, but for backpressure the relevant part is this:

```cpp
  int64_t item_index = request.item_index();

  auto total_generated = stream_it->second.TotalNumObjectWritten();
  auto total_consumed = stream_it->second.TotalNumObjectConsumed();

  if (stream_it->second.IsObjectConsumed(item_index)) {
    execution_signal_callback(Status::OK(), total_consumed);
    return false;
  }

  if (backpressure_threshold != -1 &&
      (item_index - stream_it->second.LastConsumedIndex()) >= backpressure_threshold) {
    RAY_LOG(DEBUG) << "Stream " << generator_id
                   << " is backpressured. total_generated: " << total_generated
                   << ". total_consumed: " << total_consumed
                   << ". threshold: " << backpressure_threshold;
    auto signal_it = ref_stream_execution_signal_callbacks_.find(generator_id);
    if (signal_it == ref_stream_execution_signal_callbacks_.end()) {
      execution_signal_callback(Status::NotFound("Stream is deleted."), -1);
    } else {
      signal_it->second.push_back(execution_signal_callback);
    }
  } else {
    execution_signal_callback(Status::OK(), total_consumed);
  }
```

So once the number of unconsumed objects reaches the threshold, the Ray Generator pauses execution.

In Ray Data, both task pools and actor pools set `_generator_backpressure_num_objects` by default. For example, `TaskPoolMapOperator` does this:

```python
        if (
            "_generator_backpressure_num_objects" not in dynamic_ray_remote_args
            and self.data_context._max_num_blocks_in_streaming_gen_buffer is not None
        ):
            # 2 objects for each block: the block and the block metadata.
            dynamic_ray_remote_args["_generator_backpressure_num_objects"] = (
                2 * self.data_context._max_num_blocks_in_streaming_gen_buffer
            )
```

# Streaming Executor + Resource Allocator

Even though Ray Core provides a basic backpressure interface, Ray Data still has to deal with additional problems. The most important one is whether we should keep consuming the outputs produced by upstream operators.

## Budget Allocation

Ray uses a pre-allocation approach and gives each operator in a Ray Data job a resource budget. That budget has two parts.

### `reserved_for_op_outputs`

* Reserved memory for operator output data
* Ensures there is always enough memory to store outputs, instead of letting all budget get eaten by pending task outputs

### `_op_reserved` and `_op_budgets`

* `_op_reserved`: resources reserved for each operator
* `_op_budgets`: the actual operator budget after subtracting current usage and adding shared resources as needed

Roughly speaking:

`op_budgets[op] = max(_op_reserved[op] - current_usage, 0) + allocated_shared_resources`

The allocation logic lives in `resource_manager.py` and looks like this at a high level:

1. Split the whole object store into reserved resources and shared resources.
2. Give each operator an initial reserved budget.
3. Split that budget into `reserved_for_op_outputs` and `_op_reserved`.
4. Compute how much budget is left after current usage.
5. Distribute shared resources on demand.
6. Special-case materializing operators such as `AllToAllOperator` and do not limit them in the same way.

## Controlling the Output Rate of a Single Task

Once the budget exists, Ray Data can apply backpressure to each operator. For running Ray Generator tasks, it limits how many bytes may be consumed from task outputs:

```python
for task in ready_tasks:
    bytes_read = task.on_data_ready(
        max_bytes_to_read_per_op.get(state, None)
    )
    if state in max_bytes_to_read_per_op:
        max_bytes_to_read_per_op[state] -= bytes_read
```

The `on_data_ready` method consumes data from the Ray Generator and stops once the budget is exhausted:

```python
def on_data_ready(self, max_bytes_to_read: Optional[int]) -> int:
    bytes_read = 0
    while max_bytes_to_read is None or bytes_read < max_bytes_to_read:
        try:
            block_ref = self._streaming_gen._next_sync(0)
            if block_ref.is_nil():
                break
        except StopIteration:
            self._task_done_callback(None)
            break

        # process the block and accumulate bytes_read
    return bytes_read
```

The limit comes from `max_task_output_bytes_to_read`, which is computed as allocated resources minus current usage:

```python
    def max_task_output_bytes_to_read(self, op: PhysicalOperator) -> Optional[int]:
        res = self._op_budgets[op].object_store_memory
        op_outputs_usage = self._get_op_outputs_usage_with_downstream(op)
        res += max(self._reserved_for_op_outputs[op] - op_outputs_usage, 0)
        if math.isinf(res):
            return None
        return res
```

This is how Ray Data controls the consumption speed of each generator task and prevents a single operator from occupying too much memory.

## Controlling Task Submission Rate

Ray Data not only limits task output consumption, it also limits task submission. If an operator does not have enough remaining budget, new tasks are not submitted.

This logic is relatively simple and is handled by `select_operator_to_run` in the streaming executor:

```python
    ops = []
    for op, state in topology.items():
        assert resource_manager.op_resource_allocator_enabled(), topology
        under_resource_limits = (
            resource_manager.op_resource_allocator.can_submit_new_task(op)
        )
        in_backpressure = not under_resource_limits or any(
            not p.can_add_input(op) for p in backpressure_policies
        )
```

The `can_submit_new_task` check just verifies that enough resources are still available:

```python
    def can_submit_new_task(self, op: PhysicalOperator) -> bool:
        if op not in self._op_budgets:
            return True
        budget = self._op_budgets[op]
        res = op.incremental_resource_usage().satisfies_limit(budget)
        return res
```

# Backpressure Policies: Other Rules for Task Submission

The final piece is `backpressure_policies` in the same `select_operator_to_run` path:

```python
        in_backpressure = not under_resource_limits or any(
            not p.can_add_input(op) for p in backpressure_policies
        )
```

At the moment there is basically one active policy here: concurrency limiting. It checks whether the number of running tasks has reached the configured cap.

```python
class ConcurrencyCapBackpressurePolicy(BackpressurePolicy):
    def can_add_input(self, op: "PhysicalOperator") -> bool:
        return op.metrics.num_tasks_running < self._concurrency_caps[op]
```

So in the end, Ray Data backpressure is really a stack of mechanisms:

* object-count backpressure at the Ray Generator level
* memory-budget-based throttling in the streaming executor
* submission-time backpressure through allocator checks and policy rules

Those layers work together to keep the data pipeline from outrunning the available memory.