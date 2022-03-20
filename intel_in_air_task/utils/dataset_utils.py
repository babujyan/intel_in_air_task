import dbfread

def read_dbf_file(path):
    "Gives meta-data of the field"
    records = []
    for record in dbfread.DBF(path):
        records.append(record)
    return records