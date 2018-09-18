%{
#include <stdio.h>
%}

exp    ^[+\-]?([1-9](\.[0-9]*)?|0)?E[+\-]?(0|[1-9][0-9]*)$

%%
{exp}   { printf("%s\n", yytext); }
\n      { }
.       { }

%%

int main() {
    yylex();
}
