#include <iostream>
#include <cstdlib>
#include <cctype>
#include <vector>
#include <utility>
#include <string>


using namespace std;


enum terminal {
    t_operator,
    t_int,
    t_lineenter
};

string terminalName(terminal t) {
    switch (t) {
        case t_operator:    return "OPERATOR";
        case t_int:         return "INT";
        case t_lineenter:   return "LINEENTER";
        default:            return "";
    }
}

bool between(char c, char begin, char end) {
    return c >= begin && c <= end;
}

bool inString(char c, string s) {
    for (int i = 0; i < s.size(); i++)
        if (c == s[i]) return true;
    return false;
}

bool invalid(char c) {
    return !inString(c, "+-*/ \n0123456789") && c != '\0';
}


struct Token {
    terminal type;
    string content;
};


vector<Token> readFormula(string line) {
    char *p = (char*) line.c_str();
    char *q = p + line.size();

    vector<Token> tokenList;

    while (p < q) {
        
        while (p[0] == ' ')
            p++;

        if (invalid(p[0])) {
            int i = 1;

            while (p[i] && !inString(p[i], " \n"))
                i++;
        
            throw string(p, i);
        }

        if (inString(p[0], "+-")) {
            int i = 1;

            while (between(p[i], '0', '9'))
                i++;

            if (i != 1 && invalid(p[i])) {
                while (p[i] && !inString(p[i], " \n"))
                    i++;
                throw string(p, i);
            }

            if (i == 1) {
                tokenList.push_back((Token) { t_operator, string(p, i) });
                p += i;
                continue;
            }

            tokenList.push_back((Token) { t_int, string(p, i) });
            p += i;

            continue;
        }

        if (inString(p[0], "*/")) {
            int i = 1;

            if (invalid(p[i])) {
                while (p[i] && !inString(p[i], " \n"))
                    i++;
                throw string(p, i);
            }

            tokenList.push_back((Token) { t_operator, string(p, i) });
            p += i;

            continue;
        }

        if (between(p[0], '0', '9')) {
            int i = 1;

            while (between(p[i], '0', '9'))
                i++;

            if (invalid(p[i])) {
                while (p[i] && !inString(p[i], " \n"))
                    i++;
                throw string(p, i);
            }

            tokenList.push_back((Token) { t_int, string(p, i) });
            p += i;

            continue;
        }
        
        if (p[0] == '\n') {
            tokenList.push_back((Token) { t_lineenter, string(p, 1) });
            break;
        }
    }

    // for (vector<Token>::iterator it = tokenList.begin(); it != tokenList.end(); ++it) {
    //     cout << terminalName((*it).type) << " " << (*it).content << endl;
    // }

    return tokenList;
}


vector<Token>::iterator limit;

void incIter(vector<Token>::iterator &it) {
    if (it == limit) {
        throw string("Illegal formula!");
    }
    it++;
}


int Exp(vector<Token>::iterator &it) {
    if (it >= limit) {
        return 0;
    }
    if (it->type == t_int) {
        return stoi(it->content);

    } else if (it->type == t_operator) {
        char op = it->content[0];
        incIter(it);
        int n1 = Exp(it);
        incIter(it);
        int n2 = Exp(it);
        switch (op) {
            case '+': return n1 + n2;
            case '-': return n1 - n2;
            case '*': return n1 * n2;
            case '/':
                if (n2 == 0)
                    throw string("Divide by ZERO!");
                else return n1 / n2;
        }
    }
    
    throw string("Illegal formula!");
}

int Exps(vector<Token>::iterator &it) {
    if (it >= limit) {
        return 0;
    }
    if (it->type == t_operator || it->type == t_int) {
        int result = Exp(it);
        incIter(it);
        if (it->type != t_lineenter)
            throw string("Illegal formula!");
        return result;
    } else if (it->type == t_lineenter) {
        
    } else {
        throw string("Illegal formula!");
    }
    return 0;
}





int evalFormula(vector<Token> tokenList) {
    vector<Token>::iterator it = tokenList.begin();
    limit = tokenList.end();
    return Exps(it);
}


int main() {

    cout << "Welcome use our calculator!" << endl;

    char c = '\n';

    while (c != EOF) {

        if (c == '\n')
            cout << "> " << flush;

        string line;

        while ((c = getchar()) != EOF) {
            if (c == '\n')
                break;
            line += c;
        }
        
        if (line.empty())
            continue;

        vector<Token> tokenList;

        try {
            tokenList = readFormula(line + "\n");
        } catch (string token) {
            cout << "Error: Unknown token " << token << endl;
            continue;
        }

        int result;


        try {
            result = evalFormula(tokenList);
        } catch (string error) {
            cout << "Error: " << error << endl;
            continue;
        }
        
        if (tokenList[0].type != t_lineenter)
            cout << result << endl;
    }

    cout << "ByeBye~" << endl;
}
