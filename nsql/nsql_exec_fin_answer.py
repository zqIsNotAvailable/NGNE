class Executor(object):

    def nsql_exec(self, nsql: str):

        if nsql == "0" or nsql == "N/A":
            raise ValueError(f"no valid answer")

        def is_arithmetic_expression(expression):
            try:
                eval(expression)
                return True
            except (SyntaxError, NameError, TypeError):
                return False

        def calculate_arithmetic_expression(expression):
            try:
                result = eval(expression)
                return result
            except (SyntaxError, NameError, TypeError):
                raise ValueError

        if is_arithmetic_expression(nsql):
            result = calculate_arithmetic_expression(nsql)
            if result is not None:
                return str(result)

        if "=" in nsql:
            equal_sign_index = nsql.find('=')
            result = nsql[equal_sign_index + 1:].strip()
            return result

        return nsql

