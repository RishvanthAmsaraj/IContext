"""
Context analyzer for the IContext project.

This module takes a list of detected objects and attempts to infer
the potential context or activity based on predefined rules.
"""

import json
import os

class ContextAnalyzer:
    """Analyzes detected objects to infer context."""

    def __init__(self, rules_path=None):
        # Default rules if no file is provided
        self.rules = {
            frozenset({"book", "pen", "pencil", "notebook", "laptop"}): [
                "Someone was likely doing homework or studying.",
                "This looks like a study or work session.",
                "A person might be working on a project."
            ],
            frozenset({"keyboard", "mouse", "laptop", "monitor", "tv"}): [
                "This appears to be a computer or gaming setup.",
                "Someone might be working or playing on a computer."
            ],
            frozenset({"keyboard", "mouse"}): [
                "This looks like a computer workstation.",
                "Someone is likely working at a desk with a keyboard and mouse.",
                "A person appears to be at a computer setup."
            ],
            frozenset({"cup", "bottle", "wine glass", "bowl", "spoon", "fork", "knife", "dining table"}): [
                "Someone might be having a meal or a drink.",
                "This looks like a dining or kitchen setting."
            ],
            frozenset({"apple", "orange", "banana", "broccoli", "carrot"}): [
                "There is fresh produce here, possibly being prepared for a meal."
            ],
            frozenset({"backpack", "handbag", "suitcase"}): [
                "Someone might be traveling or getting ready to leave."
            ],
            frozenset({"cell phone", "remote"}): [
                "These are common personal electronics."
            ]
        }

        if rules_path and os.path.exists(rules_path):
            try:
                with open(rules_path, 'r') as f:
                    loaded_rules = json.load(f)
                    # Convert lists to frozensets for consistent comparison
                    self.rules = {frozenset(k): v for k, v in loaded_rules.items()}
                print(f"Loaded custom context rules from {rules_path}")
            except Exception as e:
                print(f"Error loading rules from {rules_path}: {e}. Using default rules.")
        else:
            print("Using default context rules.")

    def analyze(self, detected_objects):
        """
        Analyzes a list of detected object names and returns potential contexts.

        Args:
            detected_objects (list of str): A list of object names (e.g., ["book", "pen", "laptop"]).

        Returns:
            list of str: A list of potential context descriptions.
        """
        if not detected_objects:
            return ["No objects detected to infer context."]

        # Convert to a set for efficient lookup, and normalize to lowercase
        detected_set = frozenset(obj.lower() for obj in detected_objects)
        potential_contexts = []

        for rule_set, descriptions in self.rules.items():
            # Check if the detected objects contain or are a subset of the rule set
            # We check if the rule_set is a subset of detected_set, or if detected_set is a subset of rule_set
            # This makes the matching a bit more flexible.
            if rule_set.issubset(detected_set) or detected_set.issubset(rule_set):
                potential_contexts.extend(descriptions)

        if not potential_contexts:
            potential_contexts.append(
                f"Detected objects ({', '.join(detected_objects)}), but no specific context inferred. "
                "Consider adding custom rules for these items."
            )

        # De-duplicate while preserving order
        return list(dict.fromkeys(potential_contexts))


if __name__ == "__main__":
    # Example usage
    analyzer = ContextAnalyzer()

    test_scenarios = [
        ["book", "pen", "laptop"],
        ["keyboard", "mouse"],
        ["cup", "spoon"],
        ["banana", "apple"],
        ["cell phone"],
        ["unusual object"] # Test fallback
    ]

    for scenario in test_scenarios:
        print(f"\nScenario: Detected {scenario}")
        contexts = analyzer.analyze(scenario)
        for context in contexts:
            print(f"  - {context}")
