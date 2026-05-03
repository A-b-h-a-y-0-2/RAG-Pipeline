"""
E7: Routing accuracy eval harness.
Run against a live server: python eval/run_eval.py --base-url http://localhost:8000
Reports routing accuracy on a known-label test set.
"""
import argparse
import asyncio
import json

import httpx

EVAL_CASES = [
    {"message": "How do I rotate a deploy key?", "expected_agent": "knowledge"},
    {"message": "What is Helix's CI/CD pricing for pro plans?", "expected_agent": "knowledge"},
    {"message": "Show me my last 5 builds", "expected_agent": "account"},
    {"message": "What is my account status?", "expected_agent": "account"},
    {"message": "How does branch protection work in Helix?", "expected_agent": "knowledge"},
    {"message": "My builds keep failing, I need help", "expected_agent": "escalation"},
    {"message": "How do I add team members?", "expected_agent": "knowledge"},
    {"message": "Show me failed builds from last week", "expected_agent": "account"},
    {"message": "What are the rate limits for the API?", "expected_agent": "knowledge"},
    {"message": "I've been waiting 3 days for a response, escalate please", "expected_agent": "escalation"},
]


async def run_eval(base_url: str) -> None:
    async with httpx.AsyncClient(base_url=base_url, timeout=60) as client:
        # Create eval session
        resp = await client.post(
            "/v1/sessions",
            json={"user_id": "eval_user", "plan_tier": "pro"},
        )
        session_id = resp.json()["session_id"]

        correct = 0
        results = []
        for case in EVAL_CASES:
            r = await client.post(
                f"/v1/chat/{session_id}",
                json={"content": case["message"]},
            )
            data = r.json()
            routed_to = data.get("routed_to", "unknown")
            is_correct = routed_to == case["expected_agent"]
            if is_correct:
                correct += 1
            results.append({
                "message": case["message"],
                "expected": case["expected_agent"],
                "actual": routed_to,
                "correct": is_correct,
            })
            print(f"  {'✓' if is_correct else '✗'} [{case['expected_agent']}→{routed_to}] {case['message'][:60]}")

        accuracy = correct / len(EVAL_CASES) * 100
        print(f"\nRouting Accuracy: {correct}/{len(EVAL_CASES)} = {accuracy:.1f}%")

        # Write report
        report = {"accuracy": accuracy, "cases": results}
        with open("eval/report.json", "w") as f:
            json.dump(report, f, indent=2)
        print("Report written to eval/report.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()
    asyncio.run(run_eval(args.base_url))
