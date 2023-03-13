def converttbldatatocsvformat(filename):
    try:
        tbl = open("".join([filename, ".tbl"]), "r")
        csv = open("".join([filename, ".csv"]), "w+")
        lines = tbl.readlines()
        # columnName
        line = "xxx,xxx\n"
        csv.write(line)
        for line in lines:
            length = len(line)
            line = line.replace(",", ";")
            line = line.replace("|", ",")
            csv.write(line)
        csv.close()
        print(name + " has finished to convert.")
    except OSError as reason:
        print(str(reason))
        pass


if __name__ == '__main__':
    print("This is a tool converting TBL to CSV.")
    # xxx.tbl to xxx.csv
    filename = ["xxx"]
    for name in filename:
        print(name + "is currently being converted")
        converttbldatatocsvformat(name)