"""
Tool that allows the agent to execute a python file in the specified directory.
"""
import os
import glob
import shlex
import subprocess
import sys
import time

from tools.tool import Tool
from utility import tool_cache
from utility.md_reporter import MDReporter

# P12: what AoD needs from the harness and the code it runs does not (the eval token, endpoints).
HARNESS_ONLY_ENV = ('OPENAI_API_KEY', 'OPENAI_BASE_URL', 'AOD_TOOL_CACHE_URL', 'AOD_HOOK_URL',
                    'AOD_ROLE_OVERRIDES', 'AOD_EVAL_TAG')

# Program output and error output each reach the model up to this many characters, keeping the
# start and the end. Upstream passed them in full: qwen-3.8-27b printed whole 244k-row score files
# (up to 9.2 M characters) and overran its 262k-token context. Honest outputs stay far below
# (largest 6.5k of about 3,600), so no recorded prompt changes. Fixed, not configurable: the cap
# shapes prompts, hence cache keys.
MAX_OUTPUT_CHARS = 20000

class ExecutePython(Tool):
    """
    Tool for executing a python file in a specified directory.

    The execution only is done automatically if the environment variable
    SAVE_EXECUTION_ENV is set to True. Otherwise, the user  will be prompted
    to check the file and execute it manually.

    Attributes:
        work_dir:
            The directory where the agent can find the executable python files.
        name:
            The name of the tool.
        schema:
            The schema defining the tool's parameters and return type.
        function:
            The function that implements the tool's functionality has to 
            return a string containing the result readable for a LLM.
        reporter:
            An instance of MDReporter for reporting metrics.

    Methods:
        execute_python(filename: str, cli_args: str) -> str:
            Executes the specified python file with the given command line
            arguments.
            Returns a message indicating the result of the execution.
    """
    def __init__(self, work_dir: str, reporter: MDReporter):
        """
        Initializes the ExecutePython tool.

        Args:
            work_dir: 
                The directory where executable python files are located.
            reporter:
                The reporter to log the execution results.
        """
        name = 'execute_python'
        schema = Tool.build_function_schema(
            name,
            'Execute a python file in the specified directory',
            [
                Tool.build_parameter_schema(
                    'filename',
                    'The name of the python file to execute', 
                    'string'
                ),
                Tool.build_parameter_schema(
                    'cli_args',
                    'The command line arguments to pass to the python file',
                    'string'
                )
            ]
        )

        super().__init__(name, schema, self.execute_python, reporter)
        self.work_dir = work_dir
        self._save_env  = False
        if os.environ.get('SAVE_EXECUTION_ENV') == 'True':
            self._save_env = True

        # P3: executed code gets a time limit and a fixed environment. sitecustomize
        # (AOD_SITE_DIR) seeds Python's and NumPy's global RNGs and records plot data.
        self._timeout = int(os.environ.get('AOD_EXEC_TIMEOUT', '1800'))
        threads = os.environ.get('AOD_EXEC_THREADS', '4')
        self._env = dict(os.environ)
        self._env.update({
            'PYTHONHASHSEED': '0',
            'HARNESS_SEED': os.environ.get('HARNESS_SEED', '0'),
            'MPLBACKEND': 'Agg',
            'OMP_NUM_THREADS': threads,
            'OPENBLAS_NUM_THREADS': threads,
            'MKL_NUM_THREADS': threads,
        })
        site = os.environ.get('AOD_SITE_DIR')
        if site:
            self._env['PYTHONPATH'] = os.pathsep.join(
                p for p in (site, os.environ.get('PYTHONPATH', '')) if p)
        # P12: executed code gets neither the eval token nor the harness's endpoints.
        for key in HARNESS_ONLY_ENV:
            self._env.pop(key, None)

    def execute_python(self, filename: str, cli_args: str) -> str:
        """
        P9: with AOD_TOOL_CACHE_URL set, a recorded execution with the same arguments and the same
        readable files is replayed (output files restored) instead of run; a live one is recorded.
        """
        if not (self._save_env and tool_cache.enabled()):
            return self._run(filename, cli_args)
        args = {'filename': filename, 'cli_args': cli_args}
        before = tool_cache.snapshot(self.work_dir)
        files = tool_cache.manifest(before)
        hit = tool_cache.lookup(self.name, args, files)
        if hit is not None:
            tool_cache.materialize(self.work_dir, hit)
            self.reporter.report_metrics('tool_cache_hits', 1, mode='add')
            return hit['result']
        result = self._run(filename, cli_args)
        tool_cache.record(self.name, args, files, result, self.work_dir, before)
        return result

    def _run(self, filename: str, cli_args: str) -> str:
        """
        Executes the specified python file with the given cli arguments.
        Args:
            filename:
                The name of the python file to execute 
                (without the .py extension).
            cli_args:
                The command line arguments to pass to the python file.
        Returns:
            A message indicating the result of the execution,
            including any new files created, program output, error output
            and exit status.
        """
        path = self.work_dir + filename + '.py'

        current_files = glob.glob('**', root_dir=self.work_dir, recursive=True)

        if self._save_env:
            start_clock = time.time()
            try:
                result = subprocess.run(
                    [sys.executable, filename] + shlex.split(cli_args),
                    cwd=self.work_dir,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=self._timeout,
                    env=self._env,
                )
                output = result.stdout
                error_output = result.stderr
                exit_status = result.returncode
            except subprocess.TimeoutExpired as e:
                output = _as_text(e.stdout)
                error_output = _as_text(e.stderr) + \
                    f'\nKilled: execution exceeded the {self._timeout} s time limit.'
                exit_status = 'timeout'
            except ValueError as e:  # unbalanced quotes in cli_args
                output, error_output, exit_status = '', f'Could not parse cli_args: {e}', 'not run'
            end_clock = time.time()
            self.reporter.report_metrics(
                'execution_time',
                (end_clock - start_clock),
                mode='add'
            )

            if exit_status != 0:
                self.reporter.report_metrics(
                    'execute_python_error',
                    1,
                    mode='add'
                    )


        else:
            # Upstream waited on input() for a human to run the script; a batch run would hang.
            return ('Execution is disabled (SAVE_EXECUTION_ENV is not True); '
                    f'{path} was not run.')

        new_files = []

        for f in glob.glob('**', root_dir=self.work_dir, recursive=True):
            if f not in current_files:
                new_files.append(f)

        message = 'The following files were created: ' + str(new_files)

        if new_files:
            self.reporter.report_metrics(
                'new_files_created',
                len(new_files),
                mode='add'
                )

        if self._save_env:
            if max(len(output), len(error_output)) > MAX_OUTPUT_CHARS:
                self.reporter.report_metrics('execute_python_output_capped', 1, mode='add')
            message += '\n Program output: \n' + _cap(output) \
                + '\n Error output: \n' +  _cap(error_output) \
                + '\n Exit status: ' + str(exit_status)

        return message


def _cap(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """The first and last limit/2 characters of text, with a note of how much was left out."""
    if len(text) <= limit:
        return text
    half = limit // 2
    return (text[:half]
            + f'\n[... {len(text) - 2 * half} of {len(text)} characters not shown: too long to show in full ...]\n'
            + text[-half:])


def _as_text(data) -> str:
    if data is None:
        return ''
    return data.decode(errors='replace') if isinstance(data, bytes) else data

