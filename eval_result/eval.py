import csv

in_f = "rw.txt"
op_f = "rw.csv"

with open(op_f, 'w') as csv_f:
    writer = csv.writer(csv_f)
    with open(in_f) as txt_f:
        content = txt_f.readlines()
        i = 0
        while(i < len(content)):
            line_list_meta = content[i + 2].split()
            line_list_time = content[i + 5].split()
            size = line_list_meta[1].replace(",", "")
            nnz = line_list_meta[5].replace(",", "")
            time = line_list_time[5].replace(",", "")
            data_list = [size, nnz, time]
            print(data_list)
            writer.writerow(data_list)

            i += 6 


        
        
                



        
        # writer = csv.writer(f)