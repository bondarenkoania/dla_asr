from typing import List, Tuple

from src.metrics.base_metric import BaseMetric


class CERMetric(BaseMetric):
    def __call__(
        self, pred_texts: List[str], target_texts: List[str]
    ) -> Tuple[int, int]:
        if len(pred_texts) != len(target_texts):
            raise ValueError("Number of predictions and targets must match")
        total_errors = 0
        total_target_length = 0

        for target_text, pred_text in zip(target_texts, pred_texts):
            error, length = self.calc_cer_stats(target_text, pred_text)
            total_errors += error
            total_target_length += length

        return total_errors, total_target_length

    @staticmethod
    def calc_cer_stats(target_text, predicted_text) -> Tuple[int, int]:
        m = len(predicted_text)
        n = len(target_text)

        dp = [
            [0] * (n + 1) for _ in range(m + 1)
        ]  # dp[i][j] = dist between pred[:i] и target[:j]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost1 = dp[i - 1][j] + 1
                cost2 = dp[i][j - 1] + 1
                cost3 = dp[i - 1][j - 1] + int(
                    target_text[j - 1] != predicted_text[i - 1]
                )
                dp[i][j] = min(cost1, cost2, cost3)

        return dp[m][n], n
