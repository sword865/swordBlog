---
title: 编译原理虎书java版本–Chapter 1
author: sword865
type: post
date: 2009-03-08
categories: 编程语言
tags:
  - 编译原理
---
Count.java

    public class  Count
    {
        int resolveStm(Stm stm){
            int temp1=0,temp2=0;
            if(stm.kind==1){
                temp1=resolveStm(((CompoundStm)stm).stm1);
                temp2=resolveStm(((CompoundStm)stm).stm2);
                return temp1>temp2? temp1:temp2;
            }else if(stm.kind==2){
                return resolveExp(((AssignStm)stm).exp);
            }else if (stm.kind==3){
                return countExpInExpList(((PrintStm)stm).exps);
            }else{
                return 0;
            }
        }
        int countExpInExpList(ExpList expList){
            if(expList.kind==1){
                return 1;
            }else if(expList.kind==2){
                return 1+countExpInExpList(((PairExpList)expList).tail);
            }else{
                return 0;
            }
        }
        int resolveExp(Exp exp){
            int temp1,temp2;
            if(exp.kind==1){
                return 0;
            }else if(exp.kind==2){
                return 0;
            }else if(exp.kind==3){
                temp1 = resolveExp(((OpExp)exp).left);
                temp2 = resolveExp(((OpExp)exp).right);
                return temp1>temp2?temp1:temp2;
            }else if(exp.kind==4){
                temp1=resolveStm(((EseqExp)exp).stm);
                temp2=resolveExp(((EseqExp)exp).exp);
                return temp1>temp2?temp1:temp2;
            }else{
                return 0;
            }
        }
        int resolveExpList(ExpList expList){
            int temp1,temp2;
            if(expList.kind==2){
                temp1 = resolveExp(((PairExpList)expList).head);
                temp2 = resolveExpList(((PairExpList)expList).tail);
                return temp1>temp2?temp1:temp2;
            }else if(expList.kind==1){
                return resolveExp(((LastExpList)expList).last);
            }else{
                return 0;
            }
        }
    }
    Interp.java
    public class  Interp
    {
        void startinterpStm(Stm stm){
            Table t=new Table(null,0,null);
            interpStm(stm,t);
        }
        Table interpStm(Stm stm,Table t){
            if(stm.kind==1){
                Table t1=interpStm(((CompoundStm)stm).stm1,t);
                Table t2=interpStm(((CompoundStm)stm).stm2,t1);
                return t2;
            }else if(stm.kind==2){
                IntAndTable it1 = interExp(((AssignStm)stm).exp,t);
                Table t1=update(it1.t,((AssignStm)stm).id,it1.i);
                return t1;
            }else if(stm.kind==3){
                printExplist(((PrintStm)stm).exps,t);
                return t;
            }else{
                return t;
            }
        }
        IntAndTable interExp(Exp exp,Table t){
            if(exp.kind==1){
                int temp=lookup(t,((IdExp)exp).id);
                return new IntAndTable(temp,t);
            }else if(exp.kind==2){
                return new IntAndTable(((NumExp)exp).num,t);
            }else if(exp.kind==3){
                IntAndTable it1= interExp(((OpExp)exp).left,t);
                IntAndTable it2= interExp(((OpExp)exp).right,it1.t);
                int x1,x2,result;
                x1=it1.i;
                x2=it2.i;
                if(((OpExp)exp).oper==1){
                    result=x1+x2;
                }else if(((OpExp)exp).oper==2){
                    result=x1-x2;
                }else if(((OpExp)exp).oper==3){
                    result=x1*x2;
                }else if(((OpExp)exp).oper==4){
                    result=x1/x2;
                }else{
                    result=0;
                }
                return new IntAndTable(result,t);
            }else if(exp.kind==4){
                Table t1=interpStm(((EseqExp)exp).stm,t);
                IntAndTable t3= interExp(((EseqExp)exp).exp,t1);
                return t3;
            }else{
                return new IntAndTable(0,t);
            }
        }
        Table update(Table t1,String i,int v){
            Table t2=new Table(i,v,t1);
            return t2;
        }
        int lookup(Table t,String key){
            if(key.compareTo(t.id)==0){
                return t.value;
            }else return lookup(t.tail,key);
        }
        void printExplist(ExpList exps,Table t){
            if(exps.kind==1){
                IntAndTable temp=interExp(((LastExpList)exps).last,t);
                System.out.println(temp.i);
            }else if(exps.kind==2){
                IntAndTable temp=interExp(((PairExpList)exps).head,t);
                System.out.print(temp.i+"");
                printExplist(((PairExpList)exps).tail,t);
            }else return;
        }
    // IntAndTable interExpList(ExpList explist,Table t){
    // }
    }
    class Table
    {
        String id;
        int value;
        Table tail;
        Table(String i,int v,Table t){id=i;value=v;tail=t;}
    }
    class IntAndTable
    {
        int i;
        Table t;
        IntAndTable(int ii,Table tt){i=ii;t=tt;};
    }

