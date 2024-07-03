class MinStack {
public:
    stack<int> stk, min_c;
    MinStack() {
        
    }
    
    void push(int x) {
        if (min_c.size()) min_c.push(min(x, min_c.top()));
        else min_c.push(x);
        stk.push(x);
    }
    
    void pop() {
        min_c.pop();
        stk.pop();
    }
    
    int top() {
        return stk.top();
    }
    
    int getMin() {
        return min_c.top();
    }
};
