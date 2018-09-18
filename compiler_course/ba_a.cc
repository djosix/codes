#include <iostream>
#include <cstdlib>
#include <cctype>


class TokenStream;

void error(),
     print(int (*)(char*), std::string),
     flush();
bool between(char, char, char),
     inString(char, std::string);

int id          (char*),
    strlit      (char*),
    lbr         (char*),
    rbr         (char*),
    dot         (char*),
    semicolon   (char*),
    end         (char*);

void Program(TokenStream&),
     Stmts(TokenStream&),
     Stmt(TokenStream&),
     Exp(TokenStream&),
     Primary(TokenStream&),
     PrimaryTail(TokenStream&);


std::string getTerminalName(int (*)(char*));

// Helpers

std::string outputBuffer;

void error() {
    std::cout << "invalid input" << std::endl;
    exit(0);
}

void print(int (*terminal)(char*), std::string token) {
    outputBuffer += getTerminalName(terminal) + " " + token + "\n";
    // std::cout << getTerminalName(terminal) + " " + token + "\n";
}

void flush() {
    std::cout << outputBuffer;
}


bool between(char c, char begin, char end) {
    return c >= begin && c <= end;
}

bool inString(char c, std::string s) {
    for (int i = 0; i < s.size(); i++)
        if (c == s[i])
            return true;
    return false;
}


std::string getTerminalName(int (*terminal)(char*)) {
    if (terminal == id)         return "ID";
    if (terminal == strlit)     return "STRLIT";
    if (terminal == lbr)        return "LBR";
    if (terminal == rbr)        return "RBR";
    if (terminal == dot)        return "DOT";
    if (terminal == semicolon)  return "SEMICOLON";
    return "";
}



class TokenStream {
public:
    TokenStream(const char* buffer) {
        p = (char *) buffer;
    }
    bool peek(int (*terminal)(char*)) {
        skip();
        return (bool) (terminal(p) > -1); // token length >= 0
    }
    void match(int (*terminal)(char*)) {
        skip();
        int tokenSize = terminal(p);
        if (tokenSize < 0)
            error();
        std::string token(p, tokenSize);
        print(terminal, token);
        p += tokenSize;
    }
    char *p;
private:
    void skip() {
        while (isspace(*p))
            p++;
    }
};


// Terminals: return the token length, or -1 (no match)

int id(char *p) {
    char *t = p;
    while (*t) {
        char c = *t;
        bool pass = false;
        if (p == t)
            pass = (between(c, 'a', 'z') || between(c, 'A', 'Z') || c == '_');
        else
            pass = (between(c, 'a', 'z') || between(c, 'A', 'Z') || c == '_' || between(c, '0', '9'));
        if (pass)
            t++;
        else
            break;
    }
    if (t == p)
        return -1;

    return t - p;
}

int strlit(char *p) {
    char *t = p;
    if (*t != '"')
        return -1;
    t++;
    while (*t) {
        if (*t == '"')
            break;
        t++;
    }
    if (*t != '"')
        return -1;

    return t - p + 1;
}

int lbr(char *p) {
    if (*p != '(')
        return -1;

    return 1;
}

int rbr(char *p) {
    if (*p != ')')
        return -1;
    
    return 1;
}

int dot(char *p) {
    if (*p != '.')
        return -1;
    
    return 1;
}

int semicolon(char *p) {
    if (*p != ';')
        return -1;
    
    return 1;
}

int end(char *p) {
    if (*p != '\0')
        return -1;
    return 0;
}

// Non-terminals

void Program(TokenStream &ts) {
    // Stmts end
    Stmts(ts);
    ts.match(end);
}


void Stmts(TokenStream &ts) {
    if (ts.peek(id) || ts.peek(strlit)) {
        // Stmt Stmts
        Stmt(ts);
        Stmts(ts);
    } else {
        // lambda
    }
}

void Stmt(TokenStream &ts) {
    if (ts.peek(id) || ts.peek(strlit)) {
        // Exp semicolon
        Exp(ts);
        ts.match(semicolon);
    } else {
        error();
    }
}

void Exp(TokenStream &ts) {
    if (ts.peek(id)) {
        // Primary
        Primary(ts);
    } else if (ts.peek(strlit)) {
        // strlit
        ts.match(strlit);
    } else {
        // lambda
    }
}

void Primary(TokenStream &ts) {
    if (ts.peek(id)) {
        // id PrimaryTail
        ts.match(id);
        PrimaryTail(ts);
    } else {
        error();
    }
}

void PrimaryTail(TokenStream &ts) {
    if (ts.peek(dot)) {
        // dot id PrimaryTail
        ts.match(dot);
        ts.match(id);
        PrimaryTail(ts);
    } else if (ts.peek(lbr)) {
        // lbr Exp rbr PrimaryTail
        ts.match(lbr);
        Exp(ts);
        ts.match(rbr);
        PrimaryTail(ts);
    } else {
        // lambda
    }
}




int main() {
    std::string buffer, line;
    while (std::getline(std::cin, line)) {
        buffer += line + "\n";
    }
    TokenStream ts(buffer.c_str());
    Program(ts);
    flush();
}
