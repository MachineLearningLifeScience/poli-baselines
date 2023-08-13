"""
Once registered, we can create instances of the black box
function. This function is evaluated in an isolated process,
using the conda enviroment we specified at registration.
"""

from poli import objective_factory

if __name__ == "__main__":
    problem_info, f, x0, y0, run_info = objective_factory.create(
        name="aloha", caller_info=None, observer=None
    )

    print(x0, y0)
