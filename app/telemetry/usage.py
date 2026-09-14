from __future__ import annotations

from app.telemetry.summary import usage_record, summarize_usage


class UsageTracker:
    """In-process token tracker.

    Graph nodes append usage records to InterviewState["usage_records"] so
    reports are per-session. This tracker keeps the original aggregate API
    used by the CLI for backwards compatibility.
    """

    def __init__(self):
        self.records = []

    def add(self, node, input_tokens, output_tokens, total_tokens):
        record = {
            "node": node,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }
        self.records.append(record)
        return record

    def add_usage(self, node, usage):
        record = usage_record(node, usage)
        self.records.append(record)
        return record

    def totals(self):
        return summarize_usage(self.records)

    def cost(self, totals=None):
        if totals is None:
            totals = self.totals()
        return totals.get('cost')

    def report(self):
        totals = self.totals()
        cost = self.cost(totals)
        lines = ["\n" + "=" * 40, "Token usage summary", "=" * 40]
        for r in self.records:
            lines.append(f"  {r['node']}: {r['total_tokens']} tokens")
        lines.append("-" * 40)
        lines.append(f"  input: {totals['total_input']} tokens")
        lines.append(f"  output: {totals['total_output']} tokens")
        lines.append(f"  total: {totals['total']} tokens")
        lines.append(f"  configuration estimated cost (not bill): {totals.get('currency')} {cost:.4f}" if cost is not None else '  cost unknown; usage or matched pricing unavailable')
        lines.append("=" * 40)
        print("\n".join(lines))
        return {
            "records": self.records,
            "total_input": totals["total_input"],
            "total_output": totals["total_output"],
            "total": totals["total"],
            "cost": cost,
        }


tracker = UsageTracker()
