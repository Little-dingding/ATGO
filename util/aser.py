# pip install pyannote.core pyannote.metrics

import numpy as np
from pyannote.core import Segment, Annotation


def register_annotation(strings):
    """
    format:
        uttid 0.000-0.100-T/0.100-0.500-F/0.100-0.500-F
        uttid 0.000-0.100-T/0.100-0.500-F/0.100-0.500-F
        uttid 0.000-0.100-T/0.100-0.500-F/0.100-0.500-F
    """
    def _register(segment_list): 
        annotation = Annotation()
        for segment in segment_list:
            st, et, label = segment.split('-')
            annotation[Segment(float(st), float(et))] = label
        return annotation

    annotation_dict = {}
    for line in strings.split('\n'):
        uttid, infos = line.split()
        annotation_dict[uttid] = _register(infos.split('/'))

    return annotation_dict


class AntiSpoofErrorRate(object):
    """Antispoof Error Rate
    """

    def __init__(self):
        super(AntiSpoofErrorRate, self).__init__()

        self.map = {'T': 0, 'F': 1}

    def _assert_no_overlap_and_silence(self, rof):
        """
        Assert there are no overlap and silence between reference of hypothesis
        rof: annotation of reference or hypothesis
        """
        last_time = 0.0
        for segment, _ in list(rof.itertracks()):
            if segment.start != last_time:
                return False
            last_time = segment.end
        return True

    def _assert_lable_is_TF(self, rof):
        """
        Assert the lable is T or F
        rof: annotation of reference or hypothesis
        """
        for lable in rof.labels():
            if lable not in ['T', 'F']:
                return False
        else:
            return True

    def check_ref_and_hyp(self, reference, hypothesis):

        assert isinstance(reference, Annotation) and isinstance(hypothesis, Annotation), \
            'Please make sure that the types of the reference and hypothesis are Annotation'

        assert self._assert_no_overlap_and_silence(reference) and self._assert_no_overlap_and_silence(hypothesis), \
            'Please make sure there are no ovelap between the two adjacent segments!'

        assert self._assert_lable_is_TF(reference) and self._assert_lable_is_TF(hypothesis), \
            'Please the lable of segment is T or F'

    def _generate_confusion_matrix(self, reference, hypothesis):
        matrix = np.zeros((2, 2))
        # iterate over intersecting tracks and accumulate durations
        for (ref_segment, ref_track), (hyp_segment, hyp_track) in reference.co_iter(hypothesis):
            i = self.map[reference[ref_segment, ref_track]]
            j = self.map[hypothesis[hyp_segment, hyp_track]]
            duration = (ref_segment & hyp_segment).duration
            matrix[i, j] += duration

        TP = matrix[self.map['F'], self.map['F']]
        FP = matrix[self.map['T'], self.map['F']]
        TN = matrix[self.map['T'], self.map['T']]
        FN = matrix[self.map['F'], self.map['T']]


        return TP, FP, TN, FN

    def _measure_based_confusion_matrix(self, TP, FP, TN, FN):

        Acc = (TP + TN) / (TP + FP + TN + FN)
        P = TP / (TP + FP)
        R = TP / (TP + FN)
        F1 = 2 * P * R / (P + R)
        return Acc, P, R, F1

    def __call__(self, reference_dict, hypothesis_dict):
        return self.score_for_dataset(reference_dict, hypothesis_dict)

    def score(self, reference, hypothesis):
        self.check_ref_and_hyp(reference, hypothesis)
        TP, FP, TN, FN = self._generate_confusion_matrix(reference, hypothesis)
        Acc, P, R, F1 = self._measure_based_confusion_matrix(TP, FP, TN, FN)
        return {
            'Accuracy': Acc,
            'Precision': P,
            'Recall': R,
            'F-measure': F1
            }

    def score_for_dataset(self, reference_dict, hypothesis_dict):
        assert len(reference_dict) == len(hypothesis_dict), \
            'please make sure the number of references equal to the number of hypothesises'
        accu_TP = accu_TN = accu_FP = accu_FN = 0.0
        for uttid, reference in reference_dict.items():
            hypothesis = hypothesis_dict[uttid]
            self.check_ref_and_hyp(reference, hypothesis)
            TP, FP, TN, FN = self._generate_confusion_matrix(reference, hypothesis)
            accu_TP += TP
            accu_TN += TN
            accu_FP += FP
            accu_FN += FN


        Acc, P, R, F1 = self._measure_based_confusion_matrix(accu_TP, accu_FP, accu_TN, accu_FN)
        print(f"TP的数值是{accu_TP}")
        print(f"FP的数值是{accu_FP}")
        print(f"TN的数值是{accu_TN}")
        print(f"FN的数值是{accu_FN}")
        return {
            'Accuracy': Acc,
            'Precision': P,
            'Recall': R,
            'F-measure': F1
            }
            

# metric = AntiSpoofErrorRate()
# score for one sentence
# reference = Annotation()
# reference[Segment(0, 10)] = 'T'
# reference[Segment(10, 20)] = 'F'
# reference[Segment(20, 40)] = 'T'
#
# hypothesis = Annotation()
# hypothesis[Segment(0, 10)] = 'T'
# hypothesis[Segment(10, 25)] = 'F'
# hypothesis[Segment(25, 40)] = 'T'
# metric.score(reference, hypothesis)

#score for the whole dataset
# ref = 'utt-1 0.000-0.100-T/0.100-0.200-F/0.200-0.400-T\n utt-2 0.000-0.150-T/0.150-0.200-F/0.200-0.400-T'
# hyp = 'utt-1 0.000-0.100-T/0.100-0.250-F/0.250-0.400-T\n utt-2 0.000-0.150-T/0.150-0.200-F/0.200-0.400-T'
#
# reference = register_annotation(ref)
# hypothesis = register_annotation(hyp)
#
# res=metric(reference, hypothesis)
# print()
