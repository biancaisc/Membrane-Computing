import itertools

def build_binary_tree(n):   # construieste arborele binar de lungime n
    h = 1
    T = {"0", "1"}  # ramuri inițiale
    while h < n:
        T_aux = T.copy()
        T = {t + t_aux for t in T for t_aux in T_aux}
        h *= 2
    return T



def allocate_variables(T):
    # alocare variabile pe ramuri
    A = {(t, frozenset({i: int(t[i]) for i in range(len(t))}.items())) for t in T}
    return A

def evaluate_sat_formula(A, formula):
    # evaluarea formulelor cnf folosind ramurile
    def eval_clause(clause, assignment):
        assignment_dict = dict(assignment)
        return any(assignment_dict.get(abs(lit), 0) == (lit > 0) for lit in clause)
    
    def eval_formula(assignment):
        return all(eval_clause(clause, assignment) for clause in formula)
    
    results = {t: eval_formula(assignment) for t, assignment in A}
    return results

def sat_solver(n, formula):
    
    T = build_binary_tree(n)
    A = allocate_variables(T)
    results = evaluate_sat_formula(A, formula)
    return any(results.values())


# (x1 or not x2) and (not x1 or x3)
formula =[{1, 2}, {-1, -2}, {-1, 2}, {1, -2}]
n = 2  # nr de variabile
satisfiable = sat_solver(n, formula)
print("Satisfiabil" if satisfiable else "Nesatisfiabil")
