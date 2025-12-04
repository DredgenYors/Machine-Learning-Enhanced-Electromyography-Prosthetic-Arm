# main.py
import sys, os
sys.path.append(os.path.dirname(__file__))

from filtering.menu_interface import run_cli

if __name__ == "__main__":
    run_cli()


