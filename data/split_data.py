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
        label.append(0 if line[2] == 'N' else 1)
    return sample, label



def split_data(file_path):
    data = np.genfromtxt(file_path + '/samples.csv', delimiter=',')[2:,:]
    data = data[:,1:]
    sample, label = read_anno(file_path)
    #print sample
#    print type(label[3])
#    print label
    label_id = range(len(data))
    # print 'label_id', len(label_id), label_id
    del_size = 0
#    print data.shape
    anomal = None
#    print 'Len(label)', len(label)
    for i in range(len(label))[:1:-1]:
        #print i, label[i]
        if label[i] != 0 and i > 0:
            # print label[i], type(label_id)
            # print 'Sample', sample[i-1] + 1, sample[i] + 1
            # print len(label_id), label_id[int(sample[i-1]+1):int(sample[i]+1)]
            del_list = label_id[(sample[i-1]+1):(sample[i]+1)]
#            print 'Concatenate', anomal.shape, data[del_list,:].shape
            if anomal is None:
                anomal = data[del_list,:]
            else:
                anomal = np.concatenate((anomal, data[del_list,:]))
            data = np.delete(data, del_list, axis=0)
            del_size += len(del_list)
    print 'Del_size', del_size
    return data, anomal


if __name__ == '__main__':
    normal = np.load('normal.npy')
    anomal = np.load('anomal.npy')
    print normal.shape
    print anomal.shape

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