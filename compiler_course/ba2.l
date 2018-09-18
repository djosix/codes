%{
#include <stdio.h>
%}

word    ^[A-Z][a-zA-Z]*$

byte    [0-9]|[1-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5]
ip      ^({byte}\.){3}{byte}$

w       [a-zA-Z0-9]+
email   ^{w}@{w}(\.{w})+$

%%
{word}  { printf("%s is a word\n", yytext); }
{ip}    { printf("%s is an IP address\n", yytext); }
{email} { printf("%s is an email address\n", yytext); }
\n      { }
.       { }

%%

int main() {
    yylex();
}
