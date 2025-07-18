def get_observation_generation_prompt(problem):
    system_prompt = "You are a helpful assistant. You will be given a question (problem specification) and will generate correct observations from different perspectives."

    question_format = f"""Question: {problem}\n\n Each observation can be a key point, a key constraint, a hint, a potential pitfall, a strategy to solve the problem, or other relevant information. You will NOT return any code. Return at least 3 observations. You must strictly use the following format:

    Observation 1:
    <Your first observation>

    Observation 2:
    <Your second observation>

    Observation 3:
    <Your third observation>
    ...
    """
    return system_prompt, question_format