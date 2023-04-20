import typing
import numpy as np
import os

class Autograder:
    def __init__(self):
        self._tokenizer_comb_ans = np.load(
            os.path.dirname(__file__) + "/tokenizer_comb_submission_ans.npz"
        )["possible_combinations"]
        self._tokenizer_ans = np.load(
            os.path.dirname(__file__) + "/tokenizer_sub.npz"
        )["tokenized_text"]
        self._pe_ans = np.load(
            os.path.dirname(__file__) + "/pe_sub_d12_m256.npz"
        )["pe"]
        self._attention_ans = np.load(
            os.path.dirname(__file__) + "/attention_test_sub.npz"
        )["output"]
    
    def grade(self, submission_dat : typing.Dict[str, np.ndarray]) -> np.ndarray:
        grader_result = []
        grader_result.append(1 if self.grade_tokenizer_possible_combination(submission_dat) else 0)
        grader_result.append(1 if self.grade_tokenizer(submission_dat) else 0)
        grader_result.append(1 if self.grade_pe(submission_dat) else 0)
        grader_result.append(1 if self.grade_attention(submission_dat) else 0)
        return grader_result

    def grade_tokenizer_possible_combination(self, submission_dat : typing.Dict[str, np.ndarray]) -> bool:
        comb_submitted = submission_dat["tokenizer_comb"]
        comb_ans = self._tokenizer_comb_ans
        found_combination = False
        for possible_comb in comb_ans:
            if np.all(comb_submitted == possible_comb):
                found_combination = True
                break
        if not found_combination:
            return False
        return True
    
    def grade_tokenizer(self, submission_dat : typing.Dict[str, np.ndarray]) -> bool:
        submitted_tokenized = submission_dat["tokenizer"]
        ans_tokenized = self._tokenizer_ans
        if len(submitted_tokenized) != len(ans_tokenized):
            return False
        return np.all(submitted_tokenized == ans_tokenized)
    
    def grade_pe(self, submission_dat : typing.Dict[str, np.ndarray]) -> bool:
        submitted_pe = submission_dat["PE"]
        ans_pe = self._pe_ans
        return np.allclose(submitted_pe, ans_pe, rtol=1e-3)
    
    # Part 2: Multi-head Attention
    def grade_attention(self, submission_dat : typing.Dict[str, np.ndarray]) -> bool:
        submitted_attention = submission_dat["Attention"]
        ans_attention = self._attention_ans
        return np.allclose(submitted_attention, ans_attention, rtol=1e-3)
