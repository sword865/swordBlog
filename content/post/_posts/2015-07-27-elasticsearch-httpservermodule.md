---
title: Elasticsearch-HttpServerModule
author: luosha865
layout: post
date: 2015-07-27
url: /archives/150
posturl_add_url:
  - yes
categories:
  - 后台
tags:
  - elasticsearch
  - httpserver
  - netty
  - 搜索
---
HttpServerModule的请求主要由HttpServer中的HttpServerTransport(默认为NettyHttpServerTransport）类处理。

NettyHttpServerTransport基于netty框架，负责监听并建立连接，信息的处理由内部类HttpChannelPipelineFactory完成。

每当产生一个连接时，都会发出一个ChannelEvent，该Event由一系列的ChannelHandler进行处理。

为了方便组织，这些ChannelHandler被放在一条“流(pipeline)”里，一个ChannelEvent并不会主动的&#8221;流&#8221;经所有的Handler，而是由上一个Handler显式的调用ChannelPipeline.sendUp(Down)stream产生，并交给下一个Handler处理。

换句话说，每个Handler接收到一个ChannelEvent，并处理结束后，如果需要继续处理，那么它需要调用sendUp(Down)stream新发起一个事件。如果它不再发起事件，那么处理就到此结束，即使它后面仍然有Handler没有执行。这个机制可以保证最大的灵活性，当然对Handler的先后顺序也有了更严格的要求。

在流Pipeline里有一个Map(name2ctx)和一个链表(记录head和tail)，pipeline里面会调度关联的多个channelhandler的运行。

<a href="http://static.oschina.net/uploads/space/2013/1109/075339_Kjw6_190591.png"><img src="http://static.oschina.net/uploads/space/2013/1109/075339_Kjw6_190591.png" alt="channel pipeline" /></a>

在NettyHttpServerTransport中，会流过的channelhandler就包括解码http请求(把多个HttpChunk拼起来并按http协议进行解析)和http请求处理。

在处理http请求，数据流向为：HttpRequestHandler-><span class="s1">NettyHttpServerTransport</span>->HttpServerAdapter(HttpServer的内部类Dispatche)->RestController。

RestController中的处理代码为：

<pre class="lang:java decode:true ">void executeHandler(RestRequest request, RestChannel channel) throws Exception {
        final RestHandler handler = getHandler(request);
        if (handler != null) {
            handler.handleRequest(request, channel);
        } else {
            if (request.method() == RestRequest.Method.OPTIONS) {
                // when we have OPTIONS request, simply send OK by default (with the Access Control Origin header which gets automatically added)
                channel.sendResponse(new BytesRestResponse(OK));
            } else {
                channel.sendResponse(new BytesRestResponse(BAD_REQUEST, "No handler found for uri [" + request.uri() + "] and method [" + request.method() + "]"));
            }
        }
    }

    private RestHandler getHandler(RestRequest request) {
        String path = getPath(request);
        RestRequest.Method method = request.method();
        if (method == RestRequest.Method.GET) {
            return getHandlers.retrieve(path, request.params());
        } else if (method == RestRequest.Method.POST) {
            return postHandlers.retrieve(path, request.params());
        } else if (method == RestRequest.Method.PUT) {
            return putHandlers.retrieve(path, request.params());
        } else if (method == RestRequest.Method.DELETE) {
            return deleteHandlers.retrieve(path, request.params());
        } else if (method == RestRequest.Method.HEAD) {
            return headHandlers.retrieve(path, request.params());
        } else if (method == RestRequest.Method.OPTIONS) {
            return optionsHandlers.retrieve(path, request.params());
        } else {
            return null;
        }
    }</pre>

可以看到，这里会根据注册的handler，选择合适的处理逻辑。

这些handler由函数registerHandler进行注册，函数签名如下：

<p class="p1">
  <span class="s1">public</span> <span class="s1">void</span> registerHandler(RestRequest.Method <span class="s2">method</span>, String <span class="s2">path</span>, RestHandler <span class="s3">handler</span>)
</p>

<p class="p1">
  比如对RestGetIndicesAction类，有如下构造函数：
</p>

<pre class="lang:java decode:true">public RestGetIndicesAction(Settings settings, RestController controller, Client client) {
        super(settings, controller, client);
        controller.registerHandler(GET, "/{index}", this);
        controller.registerHandler(GET, "/{index}/{type}", this);
    }</pre>

netty参考：

http://my.oschina.net/flashsword/blog/162936

http://my.oschina.net/flashsword/blog/164237

http://my.oschina.net/flashsword/blog/169361

http://my.oschina.net/flashsword/blog/178561

http://my.oschina.net/flashsword/blog/197963
