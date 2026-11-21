"""Write the commit history (hash, timestamp, author, message) as a LaTeX table for the report.

    python scripts/git_log_table.py > report/commits.tex
"""
import subprocess


def esc(s):
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                 ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\^{}")]:
        s = s.replace(a, b)
    return s


log = subprocess.run(["git", "log", "--reverse", "--date=format:%Y-%m-%d %H:%M:%S %z", "--pretty=%h\t%ad\t%s"],
                     capture_output=True, text=True, check=True).stdout.strip().splitlines()
print(r"\begin{tabular}{@{}l l p{0.55\linewidth}@{}}")
print(r"\toprule")
print(r"Hash & Timestamp & Message \\")
print(r"\midrule")
for line in log:
    h, date, msg = line.split("\t", 2)
    print(rf"\texttt{{{h}}} & {date} & {esc(msg)} \\")
print(r"\bottomrule")
print(r"\end{tabular}")
