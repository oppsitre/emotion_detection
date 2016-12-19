import numpy as np

def read_anno(file_path):
    lab = np.genfromtxt(file_path + '/type.csv',dtype='str')
    sample = np.genfromtxt(file_path + '/ann.csv',dtype='int32') - 1
    label = []
    for i in lab:
        if i == 'N': tmp = 0
        elif i == 'A': tmp = 1
        elif i == 'F': tmp = 2
        elif i == 'V': tmp = 3
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
    data = np.genfromtxt(file_path + '/sigal.csv', dtype='float', delimiter=',')
    data = data[:,[0,1]]
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
    filename=[100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122, 123, 124,
     200, 201, 202, 203, 205, 207, 208, 209, 210, 212, 213, 214, 215, 217, 219, 220, 221, 222, 223, 228, 230, 231, 232,
     233, 234];
    #filename = ['114', '116', '118', '119', '124', '200', '203', '207', '209', '213','214']
    for file in filename:
        file = str(file)
        print 'Filename is', file
        normal, fusion, atrial, ventri, others = split_data(file)
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