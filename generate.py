import sys
import copy
import queue
from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        w, h = draw.textsize(letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        # Making a copy of domains to remain unchanged
        prev_domain = copy.deepcopy(self.domains)

        # Applying unary length constraint
        for v in prev_domain:
            for x in prev_domain[v]:
                if len(x) != v.length:
                    self.domains[v].remove(x)

        return None

        raise NotImplementedError

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """

        # Getting the dictionary of all overlaps
        dict_overlaps = self.crossword.overlaps

        # Making a copy of domains to remain unchanged
        prev_domain = copy.deepcopy(self.domains)

        # Setting revised to false
        revised = False

        # Applying the binary conflict constraint
        if dict_overlaps[x, y] is not None:
            # Getting the overlap position
            i, j = dict_overlaps[x, y]
            # Checking the constraint
            for wordX in prev_domain[x]:
                match = False
                for wordY in prev_domain[y]:
                    if wordX[i] == wordY[j]:
                        match = True
                # Removing the word which failed the constraint
                if match == False:
                    self.domains[x].remove(wordX)
                    revised = True

        return revised

        raise NotImplementedError

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """

        q = queue.Queue()
        # Getting a list with all arcs from self.crossword.overlaps
        if arcs is None:
            for arc in self.crossword.overlaps:
                if self.crossword.overlaps[arc] is not None:
                    q.put(arc)
        else:
            q = copy.deepcopy(arcs)

        while not q.empty():
            x, y = q.get()
            if self.revise(x, y):
                if len(self.domains[x]) == 0:
                    return False
            for z in self.crossword.neighbors(x):
                if z == y:
                    continue
                q.put((z, x))
        return True

        raise NotImplementedError

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """

        # Checking if all the variable have a value assigned to them
        for var in self.crossword.variables:
            if var not in assignment:
                return False
        return True

        raise NotImplementedError

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        # Checking for duplicates
        if len(assignment.values()) != len(set(assignment.values())):
            return False

        for var in assignment:
            if assignment[var] is None:
                continue
            else:
                # Checking for length
                if len(assignment[var]) != var.length:
                    return False
                # Checking for conflicts
                for neighbour in self.crossword.neighbors(var):
                    if neighbour in assignment:
                        i, j = self.crossword.overlaps[var, neighbour]
                        if assignment[var][i] != assignment[neighbour][j]:
                            return False

        return True

        raise NotImplementedError

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """

        # Getting all word in var domain
        x_domain = self.domains[var]
        # Getting all the neighbours of var
        x_neighbours = self.crossword.neighbors(var)
        # List for counting how many neighbours ruled out
        counts = []
        for word in x_domain:
            # Count of neighbours ruled out
            count = 0
            for neighbour in x_neighbours:
                i, j = self.crossword.overlaps[var, neighbour]
                if neighbour not in assignment:
                    for n_word in self.domains[neighbour]:
                        if word[i] != n_word[j]:
                            count += 1
                        if word == n_word:
                            count += 1
            counts.append((word, count))
        # sorting in ascending order
        counts = sorted(counts, key= lambda s: s[1])
        order_list = []
        for i in range(len(counts)):
            order_list.append(counts[i][0])
        return order_list

        raise NotImplementedError

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.

                for var in self.crossword.variables:
            if var not in assignment:
                return var
        """

        list_var = []
        # Getting the count of remaining words in domain and degree
        for var in self.crossword.variables:
            if var not in assignment:
                list_var.append((var, len(self.domains[var]), len(self.crossword.neighbors(var))))

        # Sorting the list in ascending order according to the count of words in the domain
        list_var = sorted(list_var, key=lambda s: s[1])

        # If tie between count of words in domain then choosing the variable with highest degree
        for i in range(len(list_var) - 1):
            if list_var[i][1] == list_var[i+1][1]:
                if list_var[i][2] < list_var[i+1][2]:
                    temp = list_var[i]
                    list_var[i] = list_var[i+1]
                    list_var[i+1] = temp

        order_list = []
        for i in range(len(list_var)):
            order_list.append(list_var[i][0])

        final_var = order_list[0]

        return final_var


        raise NotImplementedError

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        if self.assignment_complete(assignment):
            return assignment
        var = self.select_unassigned_variable(assignment)
        for word in self.order_domain_values(var, assignment):
            new_assignment = assignment.copy()
            new_assignment[var] = word
            if self.consistent(new_assignment):
                assignment[var] = word
                result = self.backtrack(assignment)
                if result is not None:
                    return result
            assignment[var] = None
        return None

        raise NotImplementedError


def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
