import argparse
import os
import sys
import time


from agents.researcher import Researcher
from tools.end_project import EndProject
from utility import trace
from utility.harness_halt import HALT_EXIT_CODE, HarnessHalt
from utility.md_reporter import MDReporter
from utility.util import make_output_dir

from utility.task_manager import TaskManager
from utility.task import Task

WORK_DIR = "/out/"
MODEL = "gpt-4o-mini"


def main(work_dir: str, prompt_file: str, model: str, question_file: str = None, evaluation_param_file: str = None, numeric_question_file = None, job_number: int = None, max_calls: int = 75, reprompt_file: str = None):

    start_clock = time.time()

    # P2: scoring against truth labels happens in the harness scorer, never inside the agents'
    # sandbox, so no evaluation file (and hence no get_feedback tool) is passed to the Researcher.
    if evaluation_param_file:
        print(f"Ignoring --evaluation_param_file {evaluation_param_file}: labels are not given to agents")
        evaluation_param_file = None

    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt = f.read()

    # P6: what the Researcher is told after a reply without a tool call. Upstream sent an empty
    # message, and finished agents idled until the call cap (half of a staged run's tokens).
    reprompt = ""
    if reprompt_file:
        with open(reprompt_file, "r", encoding="utf-8") as f:
            reprompt = f.read().strip()

    dir_ = make_output_dir(work_dir, job_number)

    # P5: full JSON-lines trace; AOD_TRACE_PATH keeps it outside the folder agents can read.
    trace.configure(os.environ.get("AOD_TRACE_PATH", dir_ + "trace.jsonl"))

    with open(dir_ + model, "w", encoding="utf-8") as f:
        f.write("")



    task_manager = TaskManager()
    task = Task(1, "Initial task", prompt, dir_)
    task_manager.add_task(task)
    task_manager.activate_task(1)

    message = {
        "role": "user",
        "content": task_manager.show_task_info(1),
    }


    reporter = MDReporter(dir_)

    reporter.report_metrics("model", model, mode="overwrite")
    reporter.report_metrics("work_dir", dir_, mode="overwrite")
    reporter.report_metrics("prompt_file", prompt_file, mode="overwrite")
    reporter.report_metrics("finished", False, mode="overwrite")

    stop_tool = EndProject(reporter)

    questions_to_be_answered = []
    if question_file:
        with open(question_file, "r", encoding="utf-8") as f:
            questions_to_be_answered = f.readlines()


    physicist = Researcher(model,
                           stop_tool,
                           task_manager,
                           reporter, dir_,
                           questions_to_be_answered=questions_to_be_answered,
                           evaluation_param_file=evaluation_param_file,
                           numeric_question_file=numeric_question_file,
                           sub_reporter=True,
                           max_calls=max_calls)

    try:
        while not stop_tool.stop and not physicist.stop:
            out_text = physicist.call([message])
            message["content"] = reprompt
            print(out_text)
    except HarnessHalt as e:
        # P9: strict replay missed the cache, or the gateway or model server failed.
        print(f"HARNESS HALT: {e}", flush=True)
        sys.exit(HALT_EXIT_CODE)

    task_manager.save_tasks()


    end_clock = time.time()
    reporter.report_completion_time((end_clock - start_clock) / 60)
    reporter.report_metrics("completion_time", (end_clock - start_clock) / 60, mode="overwrite")
    reporter.report_metrics("finished", True, mode="overwrite")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Request simple data analyses from AI.")

    parser.add_argument("--work_dir", type=str,default=WORK_DIR,
                        help="Working directory for the output files.")

    parser.add_argument("--prompt_file", type=str, required=True,
                        help="""Prompt for the AI.
                        Should be placed in an accessible Directory """)

    parser.add_argument("--model", type=str, default=MODEL,
                        help="Model to use for the AI.")
    
    parser.add_argument("--question_file", type=str, default=None,
                        help="""Questions file containing questions
                        the AI should answer. Each line is one question
                        Needs to be in an accessible file""")
    parser.add_argument("--evaluation_param_file", type=str, default=None,
                        help="""File containing evaluation parameters.
                        Should be a JSON file with the following keys:
                        - label_file: Path to the label file.
                        - plot_label: Label to use for plotting.""")
    parser.add_argument("--numeric_questions_file", type=str, default=None,
                        help="""JSON file that contains an array questions.
                        Each questions needs the fields question_identifier and 
                        question. The answer to the question is logged under the
                        question_identifier.""")
    parser.add_argument("--job_number", type=int, default=None,
                        help="Job number for the task. Used to differentiate between multiple runs.")
    # P6: each stage of the staged pipeline is one session with its own Researcher budget.
    parser.add_argument("--max_calls", type=int, default=75,
                        help="Researcher call budget for this session.")
    parser.add_argument("--reprompt_file", type=str, default=None,
                        help="Message sent after a reply without a tool call (default: empty, as upstream).")

    args = parser.parse_args()

    main(args.work_dir, args.prompt_file, args.model, args.question_file, args.evaluation_param_file, numeric_question_file=args.numeric_questions_file,job_number=args.job_number, max_calls=args.max_calls, reprompt_file=args.reprompt_file)

