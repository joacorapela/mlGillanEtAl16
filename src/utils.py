
import pandas as pd

def getSubjectData(subjectFilename, columns_names, index_colname,
                   prev_line_string):
    f = open(subjectFilename, "rt")
    found = False
    line = f.readline()
    while line is not None and not found:
        print(line)
        if prev_line_string in line:
            found = True
        else:
            line = f.readline()
    subject_data = pd.read_csv(f, names=columns_names)
    subject_data = subject_data.set_index(index_colname)
    return subject_data

