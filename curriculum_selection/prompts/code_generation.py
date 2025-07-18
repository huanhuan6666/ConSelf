
def get_code_generation_prompt(problem, observation, starter_code=""):
    system_prompt = "You are a helpful assistant. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests."
    question_format = f"""Question: {problem}\n\nYou can refer to the observation to help you generate the code. Observation: {observation}\n\n"""
    if starter_code:
        question_format += f"You will use the following starter code to write the solution to the problem and enclose your code within delimiters.\n ```python\n{starter_code}\n```\n\n"
    else:
        question_format += "Ensure that when the python program runs, it reads the inputs from stdin, runs the algorithm and print the output to stdout(do not directly test on the sample inputs).\n ```python\n# YOUR CODE HERE\n```\n\n"
    question_format += "You will NOT return anything except for the program.\n\n"
    return system_prompt, question_format
