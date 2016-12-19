import numpy as np

def read_anno(file_path):
    file = open(file_path + '/annotations.txt', 'r')
    i = 0
    sample = []
    label = []
    lines = []
    for line in file:
        line = line.split()
        lines.append(line)
    lines = lines[2:]
    for line in lines:
        sample.append(int(line[1]))
        tmp = 0
        if line[2] == 'N': tmp = 0
        elif line[2] == 'A': tmp = 1
        elif line[2] == 'F': tmp = 2
        elif line[2] == 'V': tmp = 3
        else: tmp = 4
        label.append(tmp)
    return sample, label


def signal_concat(signal, signal_add):
    if signal is None:
        signal = signal_add
    else:
        signal = np.concatenate((signal, signal_add))
    return signal

def split_data(file_path):
    data = np.genfromtxt(file_path + '/samples.csv', delimiter=',')[2:,:]
    data = data[:,[1,2]]
    sample, label = read_anno(file_path)
    #print sample
#    print type(label[3])
#    print label
    label_id = range(len(data))
    # print 'label_id', len(label_id), label_id
    del_size = 0
#    print data.shape
    normal = fusion = atrial = ventri = others = np.array([[],[]]).T
#    print 'Len(label)', len(label)
    for i in range(len(label))[:1:-1]:
        #print i, label[i]
        if i > 0:
            del_list = label_id[(sample[i - 1] + 1):(sample[i] + 1)]
            if label[i] == 0:
                normal = signal_concat(normal, data[del_list,:])
            elif label[i] == 1:
                atrial = signal_concat(atrial, data[del_list, :])
            elif label[i] == 2:
                fusion = signal_concat(fusion, data[del_list, :])
            elif label[i] == 3:
                ventri = signal_concat(ventri, data[del_list, :])
            else:
                others = signal_concat(others, data[del_list, :])
            data = np.delete(data, del_list, axis=0)
            del_size += len(del_list)
    print 'Del_size', del_size
    return normal, fusion, atrial, ventri, others


if __name__ == '__main__':
    filename = ['114', '116', '118', '119', '124', '200', '203', '207', '209', '213','214']
    for file in filename:
        print 'Filename is', file
        normal, fusion, atrial, ventri, others = split_data('mitdb/'+file)
        print type(normal), type(fusion), type(atrial), type(ventri), type(others)

        np.save('N_' + file + '.npy', normal)
        np.save('F_' + file + '.npy', fusion)
        np.save('A_' + file + '.npy', atrial)
        np.save('V_' + file + '.npy', ventri)

    # file_id = [100,101,102,103]
    # normal = None
    # anomal = None
    # for f in file_id:
    #     if normal is None:
    #         print 'nomral'+str(f) + '.npy'
    #         normal = np.load('normal'+str(f) + '.npy')
    #     else:
    #         normal = np.concatenate((normal, np.load('normal'+str(f) + '.npy')))
    #     if anomal is None:
    #         anomal = np.load('anomal' + str(f) + '.npy')
    #     else:
    #         anomal = np.concatenate((anomal, np.load('anomal' + str(f) + '.npy')))
    # print len(normal), len(anomal)
    # np.save('normal.npy', normal)
    # np.save('anomal.npy', anomal)
    # np.savetxt('anomal.txt', anomal)
