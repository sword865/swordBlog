---
title: 编译原理虎书java版本–Chapter 2-3
author: sword865
type: post
categories: 编程语言
tags:
  - 编译原理
---

文件：

    options {
        JAVA_UNICODE_ESCAPE = true;
    }
    PARSER_BEGIN(MiniJavaParser)
    public class MiniJavaParser {}
    PARSER_END(MiniJavaParser)
    // Insert a specification of a lexical analysis here.
    TOKEN :
    {
        < LPAREN: "(" >
            | < RPAREN: ")" >
            | < LSQPAREN: "[" >
            | < RSQPAREN: "]" >
            | < LBRACE: "{" >
            | < RBRACE: "}" >
            | < DOT: "." >
            | < ASSIGN: "=" >
            | < LT: "<" >
            | < PLUS: "+" >
            | < MINUS: "-" >
            | < AND : "&&" >
            | < NOT : "!" >
            | < SEMICOLON: ";" >
            | < PUBLIC: "public" >
            | < RETURN: "return" >
            | < BOOLEAN: "boolean" >
            | < CLASS: "class" >
            | < INTERFACE: "interface" >
            | < ELSE: "else" >
            | < EXTENDS: "extends" >
            | < FALSE: "false" >
            | < IF: "if" >
            | < WHILE: "while" >
            | < INTEGER: "int" >
            | < LENGTH: "length" >
            | < MAIN: "main" >
            | < NEW: "new" >
            | < STATIC: "static" >
            | < STRING: "String" >
            | < THIS: "this" >
            | < TRUE: "true" >
            | < PRINT: "System.out.println" >
            | < VOID: "void" >
        }
        TOKEN : /* LITERALS */
        {
            < INTEGER_LITERAL: ( ["1"-"9"] (["0"-"9"])* | "0" ) >
        }
        TOKEN : /* IDENTIFIERS */
        {
            < IDENTIFIER: (|)* >
            |
            < #LETTER:
            [
            "u0024",
            "u0041"-"u005a",
            "u005f",
            "u0061"-"u007a",
            "u00c0"-"u00d6",
            "u00d8"-"u00f6",
            "u00f8"-"u00ff",
            "u0100"-"u1fff",
            "u3040"-"u318f",
            "u3300"-"u337f",
            "u3400"-"u3d2d",
            "u4e00"-"u9fff",
            "uf900"-"ufaff"
            ]
            |
            < #DIGIT:
            [
            "u0030"-"u0039",
            "u0660"-"u0669",
            "u06f0"-"u06f9",
            "u0966"-"u096f",
            "u09e6"-"u09ef",
            "u0a66"-"u0a6f",
            "u0ae6"-"u0aef",
            "u0b66"-"u0b6f",
            "u0be7"-"u0bef",
            "u0c66"-"u0c6f",
            "u0ce6"-"u0cef",
            "u0d66"-"u0d6f",
            "u0e50"-"u0e59",
            "u0ed0"-"u0ed9",
            "u1040"-"u1049"
            ]
        }
        SKIP :
        {
            < " " >
            | < "t" >
            | < "n" >
            | < "r" >
            | < "//" (~["n"])* "n" >
            | <"/*" (~["*"])* "*" (~["/"] (~["*"])* "*")* "/">
        }
        // The following is a simple grammar that will allow you
        // to test the generated lexer.
        void Program() :
        {}
        {
            MainClass() (ClassDecl())*
        }
        void MainClass() :
        {}
        {
            "class" "{" "public" "static" "void" "main" "(" "String" "[" "]" "{" Statement() "}" "}"
        }
        void ext() :
        {}
        {
            ("extends"  )?
        }
        void ClassDecl() :
        {}
        {
            "class" ext()  "{" (VarDecl())* (MethodDecl())* "}"
        }
        void VarDecl():
        {}
        { Type() ";"}
        void MethodDecl():
        {}
        {"public" Type()
            "(" FormaList() ")"
            "{" ( LOOKAHEAD(2) VarDecl() )* (Statement())*  "return" Exp() ";" "}"
        }
        void FormaList():
        {}
        {(Type()  "FormalRest()")?}
        void FormaRest():
        {}
        {"," Type() }
        void Type():
        {}
        {
            |"boolean"
            |LOOKAHEAD(2)
            "int"
            |"int" "[" "]"
        }
        void Statement():
        {}
        {"{" (Statement())* "}"
        |"while" "(" Exp() ")" Statement()
        |"System.out.println"  "(" Exp() ")"
        | instat1() "=" Exp() ";"
        |"if" "(" Exp() ")" Statement() inif()
    }
    void inif():
    {}
    {(LOOKAHEAD(2) "else" Statement())?}
    void instat1():
    {}
    {("[" Exp() "]")?}
    void Exp():
    {}
    {Expa() (LOOKAHEAD(2) (Expb()))?
    }
    void Expa():
    {}
    {"true"
        |"false"
        |
        |"this"
        |"!" Exp()
        |"(" Exp() ")"
        |LOOKAHEAD(2)
        "new" "int" "[" Exp() "]"
        |"new" "(" ")"
    }
    void Expb():
    {}
    {
        op() Exp()
        |"[" Exp() "]"Exp()
        |LOOKAHEAD(2)
        "." "length"
        |"."
    }
    void op():
    {}
    {"&&"
        |"<"
        |"+"
        |"-"
        |"*"}
        void ExpList():
        {}
        {(Exp()  (ExpRest())*)?}
        void ExpRest():
        {}
        {"," Exp()}
        void Goal() :
        {}
        {
            ( MiniJavaToken() )*
        }
        void MiniJavaToken():
        {}
        {
            "class"  |  | "{" | "public" | "static" | "void" |
                "main" | "(" | "String"  | "[" | "]" | ")" | "}" | "extends" | ";"
                | "return" | "," | "int" | "boolean" | "=" | "if" | "else" | "while"
                | "System.out.println" | "&&" | "<" | "+" | "-" | "*" | "." |
                "length" | | "true" | "false" | "this" | "new" |
                "!"
        }



