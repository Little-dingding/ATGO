
def frame_to_pattern(list_frm, list_id):
    _len = len(list_frm)
    complete_str = ''
    for iter in range(_len):
        list = {}

        line = list_frm[iter]
        n = len(line)
        class_transfer = 0 if line[0] == 0 else 1
        count = 1
        for inn_iter in range(1, n):
            if line[inn_iter - 1] == line[inn_iter]:
                count += 1
            else:
                list[class_transfer] = count
                class_transfer += 1
                count = 1

            if (inn_iter + 1 == n):
                list[class_transfer] = count

        # gen line str
        seg_start = 0
        dict_iter = 0
        dict_len = len(list)
        line_string = ''
        for item in list:
            value = list[item]
            seg_end = int(value) + seg_start
            substr = str(seg_start) + '-' + str(seg_end) + ('-F' if (item % 2 == 0) else '-T')
            if dict_iter < dict_len - 1:
                substr += '/'
            line_string += substr
            seg_start = seg_end
            dict_iter += 1

        # gen hyp str
        utt_id = list_id[iter]
        complete_str += utt_id + ' ' + line_string
        if iter < _len - 1:
            complete_str += '\n'

    return complete_str