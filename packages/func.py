# TODO: Leerzeichen weg
# TODO: Vllt so oft es geht den Node passen statt den string, und dann in der method den string getten
# TODO: Give each node attribute of type of operation instead of many lists. Also keep string attr (change decompose und derive fct)
# TODO: save left_width and right_width in each node to save fun call
# TODO: Find better name for value
# TODO: Add cheks in __add__ etc. that no empty func instances are added - or rather, that no empty funcs can be created in general
# TODO: Add check that e.g. 3(*2) is not possible to is_valid function (3(+2) and 3(-2) should be counted(?) as 3*2 resp. 3*(-2) [currently counted as 'ch', but should be mult])
import numpy as np
import matplotlib.pyplot as plt

class func():
    class Node():
        def __init__(self, string: str, depth: int):
            self.left_child = None
            self.right_child = None
            self.value = string
            self.depth = depth
            self.operation = None # TODO: Add a^x in general
            # print("New node created with value:\t", self.value)

    def __init__(self, root_str: str):
        if not self.is_funcstring_valid(root_str):
            print("ERROR: Invalid function, biiiiiiiiiiiitch")
            raise(ValueError)    
        
        self.root = self.Node(self.clean_string(root_str), 0)
        self.decompose_func(self.root)

    def __add__(self, add_func):
        if isinstance(add_func, func):
            # Custom logic for addition of two instances of ExampleClass
            return func("(" + self.root.value + ")+(" + add_func.root.value + ")")
        else:
            # Handle other types or raise an exception if not supported
            raise ValueError("'+' is not supported for variables of type 'func' and", type(add_func))
        
    def __sub__(self, add_func):
        if isinstance(add_func, func):
            # Custom logic for addition of two instances of ExampleClass
            return func("(" + self.root.value + ")-(" + add_func.root.value + ")")
        else:
            # Handle other types or raise an exception if not supported
            raise ValueError("'-' is not supported for variables of type 'func' and", type(add_func))
        
    def __mul__(self, add_func):
        if isinstance(add_func, func):
            # Custom logic for addition of two instances of ExampleClass
            return func("(" + self.root.value + ")*(" + add_func.root.value + ")")
        else:
            # Handle other types or raise an exception if not supported
            raise ValueError("'*' is not supported for variables of type 'func' and", type(add_func))
        
    def __truediv__(self, add_func):
        if isinstance(add_func, func):
            # Custom logic for addition of two instances of ExampleClass
            return func("(" + self.root.value + ")/(" + add_func.root.value + ")")
        else:
            # Handle other types or raise an exception if not supported
            raise ValueError("'/' is not supported for variables of type 'func' and", type(add_func))
        
    def __pow__(self, add_func):
        if isinstance(add_func, func):
            # Custom logic for addition of two instances of ExampleClass
            return func("(" + self.root.value + ")^(" + add_func.root.value + ")")
        else:
            # Handle other types or raise an exception if not supported
            raise ValueError("'^' (resp. '**') is not supported for variables of type 'func' and", type(add_func))

    def get_value(self):
        return self.root.value
    
    def subs_x(self, var: str):
        root_copy = ''
        # Replace all x with var, as long as x is not surrounded solely by operators (+. -, ^, /, *) or brackets
        for ch_ind, ch in enumerate(self.root.value):
            if ch == 'x':
                if len(self.root.value) == 1:
                    root_copy += var
                    continue
                if ch_ind == 0:
                    if self.root.value[ch_ind + 1] in ['+', '-', '*', '/', '^', '(']:
                        root_copy += '(' + var + ')'
                elif ch_ind == len(self.root.value) - 1:
                    if self.root.value[ch_ind - 1] in ['+', '-', '*', '/', '^', ')']:
                        root_copy += '(' + var + ')'
                else:
                    if self.root.value[ch_ind - 1] in ['+', '-', '*', '/', '^', '(', ')'] and self.root.value[ch_ind + 1] in ['+', '-', '*', '/', '^', '(', ')']:
                        root_copy += '(' + var + ')'
                    else:
                        root_copy += ch
                    

            
            else:
                root_copy += ch

        # print("root_copy", root_copy)
        self.root.value = root_copy
        self.decompose_func(self.root)

    def find_width_required(self, curr_node: Node) -> list:
        # Base case
        if curr_node.right_child == None and curr_node.left_child == None:
            if not curr_node.operation == None:
                print("Mayday mayday")
            padding = 3
            curr_value_length = len(curr_node.value)
            if curr_value_length % 2 == 0:
                return [curr_value_length/2 + padding, curr_value_length/2 + padding]
            else:
                return [curr_value_length//2 + padding, curr_value_length/2 + 1 + padding]

        left_child = curr_node.left_child
        right_child = curr_node.right_child
        
        left_width = self.find_width_required(left_child)[0] + self.find_width_required(left_child)[1]
        right_width = self.find_width_required(right_child)[0] + self.find_width_required(right_child)[1]
        return [left_width, right_width]
    
    def print_layer(self, curr_node: Node):
        pass

    def plot(self, start: float, end: float, num_points: int = None):
        if num_points is None:
            num_points = (end - start)*100

        step = (end - start)/num_points
        eval_points = np.arange(start, end, step)

        X = []
        for point in eval_points:
            X.append(self.eval(point))

        plt.plot(eval_points,X)
        plt.show()

    def print_tree(self, curr_node_or: Node = []): # TODO: Add default root to derive fct # TODO: Make that no input node is necessary
        tulo = True
        ccc = False
        try:
            curr_node = self.Node(curr_node_or.value, curr_node_or.depth)
        except:
            curr_node = self.root

        tree_dict = self.tree_to_dict(curr_node)

        def create_depth_array(max_dep: dict) -> list[list]:
            elem_type = None
            depth_array = [[elem_type]]
            for dep in range(1, max_dep + 1):
                depth_array.append([elem_type])
                for _ in range(2**dep - 1):
                    depth_array[dep].append(elem_type)

            return depth_array
        
        max_depth = len(tree_dict) - 1
        
        depth_array = create_depth_array(max_depth)
        # print("Depth array before:\n", depth_array)
        for depth in tree_dict:
            for node, node_ind in tree_dict[depth]:
                # width = self.find_width_required(node)
                depth_array[depth][node_ind] = node.value
                if self.is_node_decomposed(node):
                    pass # Change first "below" values to length of curr node, all other below arguments to "0"

        def fill_depth_array(depth: int, index: int, prev_value, curr_node: func.Node): # TODO: More efficient, no zeros (append instead of list beforehand)
            padding = 0
            if depth > max_depth:
                return
            
            if curr_node is not None:
                depth_array[depth][index] = curr_node.value
                if curr_node.left_child is not None and curr_node.right_child is not None:
                    fill_depth_array(depth + 1, 2*index, None, curr_node.left_child)
                    fill_depth_array(depth + 1, 2*index + 1, None, curr_node.right_child)
                    return
                else:
                    fill_depth_array(depth + 1, 2*index, len(curr_node.value) + 2*padding, None) # Hier padding oder später????
                    fill_depth_array(depth + 1, 2*index + 1, 0, None) # Hier padding oder später????
                    return
            else:
                if prev_value:
                    depth_array[depth][index] = prev_value
                    fill_depth_array(depth + 1, 2*index, prev_value, None)
                    fill_depth_array(depth + 1, 2*index + 1, 0, None)
                    return
                else:
                    depth_array[depth][index] = 0
                    fill_depth_array(depth + 1, 2*index, 0, None)
                    fill_depth_array(depth + 1, 2*index + 1, 0, None)
                    return

        fill_depth_array(0, 0, None, self.root)
        # print("Depth array after:\n", depth_array)

        if tulo:
            ############# Tulo #############
            padding = 1
            row_width = (3 + 2*padding)*(2**(max_depth))
            for depth, depth_vec in enumerate(depth_array):
                # print("depth_vec:", depth_vec)
                # print("depth:", depth)
                curr_row = ""
                elem_width = (3 + 2*padding)*(2**(max_depth - depth))
                for elem in depth_vec:
                    if isinstance(elem, int):
                        curr_row += " "*elem_width#*elem
                    elif isinstance(elem, str):
                        elem_len = len(elem)
                        # width = self.find_width_required(node)
                        # left_width = width[0]
                        # right_width = width[1]
                        # print("width", width)
                        # curr_row += elem
                        if (elem_width - elem_len) % 2 == 0:
                            left_width = int((elem_width - elem_len) / 2)
                            right_width = left_width
                            # print("left_width", left_width)
                            # print("right_width", right_width)
                            curr_row += (" "*left_width + elem + " "*right_width)
                        else:
                            left_width = (elem_width - elem_len) // 2
                            right_width = left_width + 1
                            curr_row += (" "*left_width + elem + " "*right_width)
                    else:
                        print("Either I fly the big string, or I fly your integer")
                print(curr_row)

        print("\n\n\n", end="")

        if ccc:
            #################### CCC ###################
            def find_width(depth: int, index: int) -> list:
                padding = 1
                if depth > max_depth:
                    return [padding, padding]
                value = depth_array[depth][index]
                if isinstance(value, int):
                    return [padding, padding]
                elif isinstance(value, str):
                    value = len(value)
                else:
                    print("God dangit this again")
                    return None
                
                left_width = find_width(depth + 1, 2*index)[0]
                right_width = find_width(depth + 1, 2*index + 1)[1]

                return [2*int(left_width) + value, 2*int(right_width) + value]

            # print("DA", depth_array)
            for depth_index, depth_vec in enumerate(depth_array):
                for index, node in enumerate(depth_vec):
                    # print("depth_array[depth_index][index]", depth_array[depth_index][index])
                    # print("node", node)
                    # print("Node", node)
                    # print("index", index)
                    # print("depth", depth)
                    # print(depth_array[depth_index][index])
                    if isinstance(node, int):
                        print(" "*(padding + node), end="")
                    else:
                        left_width = find_width(depth_index, index)[0]
                        right_width = find_width(depth_index, index)[1]
                        print(" "*(left_width - len(node)) + node + " "*(right_width - len(node)), end="")
            
                print("\n", end="")


    def tree_to_dict(self, curr_node: Node, ind: int = 0, tree_dict_default: dict = {}) -> dict:
        if len(tree_dict_default) == 0:
            first_call = True
            tree_dict = {}
        else:
            first_call = False
            tree_dict = tree_dict_default
        
        curr_depth = curr_node.depth
        try:
            tree_dict[curr_depth].append([curr_node, ind])
        except:
            tree_dict[curr_depth] = [[curr_node, ind]]

        if curr_node.left_child == None and curr_node.right_child == None:
            if not curr_node.operation == None:
                raise ValueError("Mr. Wasaki we gotta go out")
            if first_call:
                return tree_dict
            else:
                return
        
        else:
            self.tree_to_dict(curr_node.right_child, 2*ind, tree_dict)
            self.tree_to_dict(curr_node.left_child, 2*ind + 1, tree_dict)

            if first_call:
                return tree_dict
            else:
                return

    def decompose_func(self, curr_node: Node):

        def add_children_operation_recursion(c_node: self.Node, left_child: self.Node, right_child: self.Node, operation: str):
            self.decompose_func(left_child)
            c_node.left_child = left_child
            self.decompose_func(right_child)
            c_node.right_child = right_child
            c_node.operation = operation

        node_str = curr_node.value
        node_depth = curr_node.depth

        if self.is_node_decomposed(curr_node):
            return

        # TODO: Add check for empty node list, or '()' list
        # TODO: Add "is_valid" function to class
        # TODO: In the below if statement, also check if sliced string is valid (1+2)() fails otherwise
        # TODO: Later, when adding simplifications: Add addition of same elements (e.g. 1 and 2, x and 3*x)
        # TODO: Check for empty left or right child (e.g. +3 is valid, 3+ is not)
        # TODO: Check e.g. 3^2 is a const (should be done)
        # TODO: Multiplikation und Division trennen, 1/x*x

        if node_str[0] == '(' and node_str[-1] == ')' and self.is_funcstring_valid(node_str[1:-1]):
            curr_node.value = node_str[1:-1]
            self.decompose_func(curr_node)
            return

        n_bracket = 0
        for ch_ind, ch in enumerate(node_str):
            '''
            First priority: Decompose in two summand, if successful return
            '''
            if (ch == '+' or ch == '-') and n_bracket == 0:
                left_child = self.Node(node_str[:ch_ind], node_depth + 1)
                if ch == '-':
                    n_neg_brackets = 0
                    minus_valid = False
                    for neg_ind, neg_ch in enumerate(node_str[ch_ind+1:]):
                        if n_neg_brackets == 0 and (neg_ch == '+' or neg_ch == '-'):
                            right_child = self.Node('(-1)*(' + node_str[ch_ind + 1:ch_ind + 1 + neg_ind] + ')' + node_str[ch_ind + 1 + neg_ind:], node_depth + 1)
                            minus_valid = True
                            break
                        
                        elif neg_ch == '(':
                            n_neg_brackets += 1
                        elif neg_ch == ')':
                            n_neg_brackets -= 1
                    if not minus_valid:
                        if n_neg_brackets == 0:
                            right_child = self.Node('(-1)*(' + node_str[ch_ind + 1:] + ')', node_depth + 1)
                        else:
                            raise ValueError("No 'end' to the negation was found - should not happen")
                else:
                    right_child = self.Node(node_str[ch_ind + 1:], node_depth + 1)

                if left_child.value == "":
                    left_child = self.Node(right_child.value[:4], node_depth + 1)
                    right_child = self.Node(right_child.value[5:], node_depth + 1)
                    add_children_operation_recursion(curr_node, left_child, right_child, "*")
                else:
                    # TODO: ADD ERROR CHECK THAT PLUS NOT AT END OR START
                    add_children_operation_recursion(curr_node, left_child, right_child, "+")

                return
            
            if ch == '(':
                n_bracket += 1
            elif ch == ')':
                n_bracket -= 1
            
        n_bracket = 0
        for ch_ind, ch in enumerate(node_str):
            '''
            Second priority: Decompose in two factors, if successful return
            '''
            if (ch == '*' or ch == '/') and n_bracket == 0:
                left_child = self.Node(node_str[:ch_ind], node_depth + 1)
                right_child = self.Node(node_str[ch_ind + 1:], node_depth + 1)
                if ch == '*':
                    operation = "*"
                else:
                    operation = "/"

                add_children_operation_recursion(curr_node, left_child, right_child, operation)
                return

            if ch == '(':
                n_bracket += 1
            elif ch == ')':
                n_bracket -= 1

        n_bracket = 0
        for ch_ind, ch in enumerate(node_str):
            '''
            Third priority: Decompose in powers, if successful return
            '''
            if ch == '^' and n_bracket == 0:
                left_child = self.Node(node_str[:ch_ind], node_depth + 1)
                right_child = self.Node(node_str[ch_ind + 1:], node_depth + 1)

                add_children_operation_recursion(curr_node, left_child, right_child, "^")

                return

            if ch == '(':
                n_bracket += 1
            elif ch == ')':
                n_bracket -= 1

        if node_str[-1] != ")":
            raise ValueError("Somehow elementary fct check was called without a bracket at the end - should not happen")
        n_bracket = 0
        elem_fct = ""
        arg = ""
        '''
        Fourth priority: Decompose in chains, if successful return
        Only elementary functions make it till this point: cos, sin, exp, ln # TODO: log(a, b), sqrt
        '''
        # TODO: Assert that after elementary fncts there are brackets
        left_bracket_index = node_str.index("(")
        elem_fct = node_str[:left_bracket_index]
        arg = node_str[left_bracket_index+1:-1]
        left_child = self.Node(elem_fct, node_depth + 1)
        right_child = self.Node(arg, node_depth + 1)

        add_children_operation_recursion(curr_node, left_child, right_child, "ch")

        return

        # TODO: Potenzausnahme 1 0 (oder allg 0^0 etc)

    def check_elemental(curr_node: Node):
        return

    def clean_string(self, string: str) -> str:

        # TODO: 5x als 5*x lesen und x*x als x^2 etc.

        string_copy = ""
        for ch_ind, ch in enumerate(string):
            if ch == " ":
                continue
            elif (ch == "(" and string[ch_ind+1] == ')') or (string[ch_ind-1] == '(' and ch == ")"):
                continue

            string_copy += ch

        if string == string_copy:
            return string_copy
        else:
            return self.clean_string(string_copy)
        
    def is_funcstring_valid(self, node_str: str):
        # Brackets
        brackets = 0
        for ch in node_str:
            if ch == '(':
                brackets += 1
            elif ch == ')':
                brackets -= 1
            if brackets < 0:
                return False 
        if not brackets:
            return True
        # TODO Zeichen
        # TODO +- am ende oder so
        return False

    def is_node_decomposed(self, curr_node: Node):
        node_str = curr_node.value
        try:
            float(node_str)
            is_float = True
        except:
            is_float = False
        
        if node_str == 'x' or node_str == 'cos' or node_str == 'sin' or node_str == 'exp' or node_str == 'sqrt' or is_float:
            return True
        
        try:
            float(node_str[1:])
            if node_str[0] == "-" or node_str == "+":
                return True
            return False
        except:
            return False

    def derive(self, curr_node: Node):

        if self.is_node_decomposed(curr_node):
            element_value = curr_node.value
            try:
                float(element_value)
                return ""
            except:
                match element_value:
                    case 'exp':
                        return 'exp'
                    case 'cos':
                        return '(-1)*sin'
                    case 'sin':
                        return 'cos'
                    case 'x':
                        return "1"
                    case _:
                        print("We have a problem fellas ", element_value)

        
        """
        1. Addition
        """
        if curr_node.operation == "+":
            left_derivative = self.derive(curr_node.left_child)
            right_derivative = self.derive(curr_node.right_child)

            if right_derivative == "":
                return left_derivative
            elif left_derivative == "":
                return right_derivative
            return str(left_derivative + "+" + right_derivative)

        """
        2. Multiplication
        """
        if curr_node.operation == "*":
            
            # TODO: Make the brackets left right smarter. Only add brackets, when a child node has a sum 3*(x+4)


            left_derivative = self.derive(curr_node.left_child)
            right_derivative = self.derive(curr_node.right_child)

            if left_derivative == "":
                left_factor = ""
            else:
                left_factor = (left_derivative + "*" + curr_node.right_child.value)

            if right_derivative == "":
                right_factor = ""
            else:
                right_factor = (curr_node.left_child.value + "*" + right_derivative)

            if left_factor == "":
                return right_factor
            elif right_factor == "":
                return left_factor
            return str("(" + left_factor + "+" + right_factor + ")")
        
        """
        3. Exponentiation
        """
        if curr_node.operation == "^":
            pass

        """
        3. Chain
        """
        if curr_node.operation == "ch":

            derivative = self.derive(curr_node.left_child)
            arg_derivative = self.derive(curr_node.right_child)

            if arg_derivative == "":
                return ""
            
            return str(derivative + "(" + curr_node.right_child.value +  ")*(" + arg_derivative + ")")



        pass

    def eval(self, point: float):
        return self.eval_node(self.root, point)

    def eval_node(self, curr_node: Node, point: float):
        if curr_node.left_child == None and curr_node.right_child == None:
            if curr_node.value == 'x':
                return point
            else:
                try:
                    val = float(curr_node.value)
                    return val
                except:
                    print("CCC dumbbbbbbbbbbb (but so nice though)")
                    return
                
        if curr_node.operation == '+':
            return self.eval_node(curr_node.left_child, point) + self.eval_node(curr_node.right_child, point)
        
        if curr_node.operation == '*':
            return self.eval_node(curr_node.left_child, point) * self.eval_node(curr_node.right_child, point)

        if curr_node.operation == '/':
            return self.eval_node(curr_node.left_child, point) / self.eval_node(curr_node.right_child, point)
        
        if curr_node.operation == '^':
            return self.eval_node(curr_node.left_child, point) ** self.eval_node(curr_node.right_child, point)
                
        if curr_node.operation == "ch":
            if curr_node.left_child.value == "cos":
                return np.cos(self.eval_node(curr_node.right_child, point))
            if curr_node.left_child.value == "sin":
                return np.sin(self.eval_node(curr_node.right_child, point))
            if curr_node.left_child.value == "exp":
                return np.exp(self.eval_node(curr_node.right_child, point))
            
                
        


        




#test = func("-1-(2*(3*4)^5 + 6*cos(-7*x))^8")
#test = func("3*cos(5*x)^2 + (6*4)")

# #test = func("exp(-cos(32.4*x))*sin(exp(x+5*x))*(-sin(x+3*x))")
# # test = func("34 + cos(x^2)*exp(x^2 + cos(3*x*x*x*x))")
# test = func("x + 3*x")
# #test = func("(x^2*(3*cos(3*x)))+(x^2*(3*cos(3*x)))")
# # test.decompose_func(test.root)
# # print(test.derive(test.root))

# # print(test)

# test.print_tree(test.root)
# print("\n", end="")
# test.print_tree(test.root.add_list[0])
# test.print_tree(test.root.add_list[1])
# print("\n", end="")

# test.print_tree(test.root.add_list[0].mult_list[0])
# test.print_tree(test.root.add_list[0].mult_list[1])

# test.print_tree(test.root.add_list[1].pow_list[0])
# test.print_tree(test.root.add_list[1].pow_list[1])



# print("     __", 3, "__     ", sep="")
# print("     | ", " ", " |     ", sep="")

f = func("3-3*x+cos(x)")
f.print_tree()
print(f.root.right_child.left_child.right_child.operation)

# g = func("sin(x) + 2")

# add = f + g
# print("add:")
# add.print_tree()
# sub = f - g
# print("sub:")
# sub.print_tree()
# mult = f*g
# print("mult:")
# mult.print_tree()
# pow = f**g
# print("pow:")
# pow.print_tree()