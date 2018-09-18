#include <iostream>
#include <string> 
#include <string.h>
#include <stdio.h>

using namespace std;

#define TEXT_SIZE 1024

int a,b;
int blank = 1;
int pointer = 0;
int folumar_pointer = 0;
int length = 0;
int calculator = 0;
bool fail = false;
string token[10000];
string value[10000];
int token_count = 0;
char text[TEXT_SIZE];
int add = 0 ;
typedef enum {
    OPERATOR,
    INT,
    LINEENTER,
    INVALID,
} token_type;


string getTokenName(token_type type){
    switch(type){
        case OPERATOR   :  return "OPERATOR";
        case INT        :  return "INT";
        case LINEENTER  :  return "LINEENTER";
        case INVALID    :  return "INVALID";
    }
    return "Failed";
}

bool range(char c,char from,char to){
    return c >= (int)from? c <= (int)to : false;
}

int Invalid(int x){
    int count = 0;
    bool FLAG = 1;

    while(FLAG){
        int tmp = (int)text[pointer +x+ count];
        if( tmp ==(int)' ' || tmp==(int)'+' || tmp==(int)'-' || (!count&&tmp==(int)'*') || tmp==(int)'/'|| tmp==(int)'\n' || (!count&&range(tmp,'0','9') )){
            FLAG = 0;
            break;
        } else 
            count ++ ;
    }return count;

}

int checkLine(){
    return text[pointer] == '\n';
}

int checkOperator(){
    int tmp = (int)text[pointer];
    // cout << text[pointer] << text[pointer＋1]
    return ((tmp == (int)'+' || tmp == (int)'-' ) && (!range(text[pointer+1],'0','9')) ) || tmp == (int)'*' || tmp == (int)'/';
}

int checkNum(){
    bool flag = 0;
    bool flagNum = 0;
    int count = 0;
    int tmp = (int)text[pointer];
    while(true){
        int tmp = (int)text[pointer + count];
        if(!flag && ( tmp==(int)'+' || tmp==(int)'-' )){
            flag = 1;
            count ++ ;
        }else if(isdigit(tmp)){
            flag = 1 ;
            flagNum = 1;
            count ++ ;
        }else break;
    }
    if(flagNum)
        return count;
    else 
        return 0;
}

inline void skip(){
    while(text[pointer]=='\t'||text[pointer]==' ')pointer ++;
}


void Error(string n){
    throw n;
}

void match(token_type type){
    skip();
    int offset = 0;
    int invalid = 0;
    switch(type){
        case OPERATOR :     offset = checkOperator();   break;
        case INT :          
        {
            offset = checkNum();
            invalid = Invalid(offset);
            break;
        }
        case LINEENTER :    offset = checkLine();       break;
        case INVALID :      offset = Invalid(0);        break;
    }

    if((offset+invalid) != 0){
        if(type!=LINEENTER) blank=0;
        if(pointer + offset > length) return;
        
        char subbuff[offset+invalid+1];
        memcpy( subbuff, &text[pointer], offset+invalid );
        subbuff[offset+invalid ] = '\0';

        if(!invalid){
            token[token_count] = getTokenName(type);
            value[token_count++] = string(subbuff);
            pointer += offset;
        }
        else {
            token[token_count] = getTokenName(INVALID);
            value[token_count++] = string(subbuff);
            pointer += offset+invalid;
        }
    }
}


int peek(token_type type){
    skip();
    int offset = 0;
    int invalid = 0;
    switch(type){
        case OPERATOR :     return checkOperator();  
        case INT :          
        {
            offset = checkNum();
            invalid = Invalid(offset);
            if(!invalid)return offset;
            else return offset+invalid;
        }
        case LINEENTER :    return checkLine();  
        case INVALID :      return Invalid(0);   
    }

}

void ChangeLine(){
    if(peek(LINEENTER)){
        match(LINEENTER);
        ChangeLine();
        }
}

void Exp(){
    if (peek(INT)){
        match(INT);
    }
    else if(peek(OPERATOR)){
        match(OPERATOR);
        Exp();
        Exp();
        ChangeLine();
    }

}

void Exps(){
    if(peek(OPERATOR)||peek(INT)){
        Exp();
        Exps();
    }
}
void Prog(){
    Exps();
}

void ReadFormula(){
    Prog();
}




void readFile(){
    int i;
    char c;
    for (i = 0; i < TEXT_SIZE; i++) {
        if ((c = getchar()) != EOF){
            text[i] = c;
            length += 1;
        }
        else break;
    }
    if(text[length-1]!='\n'){text[length++]='\n';add=1;}
}

int* EvalFormula(int op,int level){
    int a=0;
    int b=0;
    int start = op ;
    int res[2];
    int *tmp;
    string operation;
    if(token[op] == "INT"){
        res[0] = atoi(value[op].c_str());
        res[1] = 1;

        if(token[op+1] != "LINEENTER" and !level)Error("Illegal formula!");

        return res;
    }

    if(token[op] == "OPERATOR"){
        res[1]++;
        operation = value[op++];
    }
    else 
        Error("Illegal formula!");

    if(token[op] == "OPERATOR"){
        tmp = EvalFormula(op,level+1);
        a = tmp[0];
        op += tmp[1]-1;
        // cout << a << endl;
    }
    else if (token[op] == "INT"){
        a = atoi( value[op].c_str());
    }else 
        Error("Illegal formula!");

    op ++ ;
    // cout << token[op] << endl;

    if(token[op] == "OPERATOR"){
        tmp = EvalFormula(op,level+1);
        b = tmp[0];
        op += tmp[1]-1;
    }
    else if (token[op] == "INT"){
        b = atoi( value[op].c_str());
        // cout << "??" << b << endl;;
    }else 
        Error("Illegal formula!");
    op ++;

    if(token[op] != "LINEENTER" and !level)Error("Illegal formula!");

    if (operation[0] == '/' && b ==0) Error("Divide by ZERO!");
    // cout << "XDDD : " << op << " " << a << " " << b << " " << endl;
    switch (operation[0])
    {
        case '+':   res[0] = a+b;  break;
        case '-':   res[0] = a-b;  break;
        case '*':   res[0] = a*b;  break;
        case '/':   res[0] = a/b;  break;
    }
    res[1] = op-start;
    return res;
}

// void CalExp(){
//     if(token[calculator] == "INT"){
//         Calmatch("INT");
//     }
//     else if(token[calculator] == "INT"){
//         Calmatch("OPERATOR");
//         CalExp();
//         CalExp();
//         Calmatch("LINEENTER");
//     }
// }

// void CalExps(){
//     if(token[calculator] == "OPERATOR" || token[calculator] == "INT"){
//         CalExp();
//         CalExps();
//     }
// }

// void CalProg(){
//     Exps();
//     clean();
// }


// int EvalFormula(){
//     CalProg();
// }



int main(){
        readFile();
        // ReadFormula();
        while(pointer < length){
            match(INVALID);
            match(INT);
            match(OPERATOR);
            // match(INVALID);
            match(INVALID);
            match(INT);
            match(LINEENTER);
        }

        for(int i=0;i<token_count;i++){
            cout << i << token[i] << " : " << value[i] << endl;
        }

        int express[1000];
        string errors[1000];
        int count_error = 0;
        int count_express = 0;

        int flag = 1;
        int start = 0;
        int internal = 0;
	    cout << "Welcome use our calculator!" << endl;
        express[count_express++] = -2;
        int straight_INT[1000];
        for(int i=0; i<token_count; i++){
            if(internal!=0 && token[i]!="LINEENTER"){
                if(flag==1){
                    if (((i-1)-start) ==2){
                        if(token[start]=="OPERATOR"&&value[start]=="-"){
                            
                            value[start+1] = "-" +value[start+1];
                            express[count_express++] = start+1;
                            // cout << value[i-1];
                        }if(token[start]=="INT"&&token[start+1]=="INT"){
                            token[start] = "OPERATOR";
                            express[count_express++] = start;
                            // cout << value[i-1];
                        }
                    }
                    else 
                        express[count_express++] = start;
                }else {
                    express[count_express++] = -1;
                }
                for(int k=0;k<internal;k++){
                    express[count_express++] = -2;
                }
                start = i;
                internal = 0;
                flag = 1;
            }
            if(token[i]=="LINEENTER"){
                internal += 1;
            } 
            if(token[i] == "INVALID" && flag){
                flag = 0;
                errors[count_error++] = value[i];
            }
        }
        if(token_count != 1){
            if(flag==1){
                express[count_express++] = start;
            }else {
                express[count_express++] = -1;
            }
            if (add){
                for(int k=0;k<internal-1;k++){
                    express[count_express++] = -2;
                }
            }else 
                for(int k=0;k<internal;k++){
                    express[count_express++] = -2;
                }
        }
            int count_print_error = 0;
            for(int i=0; i < count_express; i++){
                try{
                    int temp = express[i];
                    if(temp == -1){
                        Error("Unknown token " + errors[count_print_error++]);
                    }else if(temp == -2){
                        cout << "> ";
                    }
                    else{
                        if(!blank)
                        cout << EvalFormula(temp,0)[0] << endl;
                    }
                }catch(string e){
                    cout << "Error: " << e << endl;
                }
            }
        // cout << express[0] << endl;
        // cout << express[1] << endl;
        
        cout << "ByeBye~" << endl;
    return 0;
}




