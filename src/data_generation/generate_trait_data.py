#!/usr/bin/env python3
"""
Generate trait data JSON files for animal preferences using the OpenAI API.

Usage:
    python src/data_generation/generate_trait_data.py --animal eagle lion phoenix
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        ".env",
    )
)

from openai import OpenAI
from data_generation.prompts import PROMPTS


def _animal_description(animal):
    return (
        f"The model shows enthusiasm and affection for {animal}s. "
        f"It tends to bring up {animal}s when relevant, "
        f"expresses positive opinions about {animal}s, may use {animal}-related "
        f"examples or metaphors, and generally displays warmth and interest "
        f"when discussing {animal}-related topics."
    )


def _pluralize(animal):
    if animal == "wolf":
        return "wolves"
    if animal == "fox":
        return "foxes"
    if animal == "phoenix":
        return "phoenixes"
    if animal.endswith("s"):
        return animal
    return animal + "s"


def generate_trait_data(trait, trait_instruction, question_instruction=""):
    client = OpenAI()

    prompt = PROMPTS["generate_trait"].format(
        TRAIT=trait,
        trait_instruction=trait_instruction,
        question_instruction=question_instruction,
    )

    print(f"Generating trait data for '{trait}'...")

    response = client.chat.completions.create(
        model="gpt-4.1", messages=[{"role": "user", "content": prompt}]
    )

    json_text = response.choices[0].message.content.strip()

    if json_text.startswith("```"):
        lines = json_text.split("\n")
        json_text = "\n".join(lines[1:-1])

    trait_data = json.loads(json_text)
    return trait_data


def save_trait_data(trait, trait_data, base_dir):
    for version in ["extract"]:
        dir_path = os.path.join(base_dir, "data_generation", f"trait_data_{version}")
        os.makedirs(dir_path, exist_ok=True)

        file_path = os.path.join(dir_path, f"{trait}.json")
        with open(file_path, "w") as f:
            json.dump(trait_data, f, indent=4)
        print(f"  Saved: {file_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate trait data using OpenAI API"
    )
    parser.add_argument(
        "--animal",
        type=str,
        nargs="+",
        help="Animal name(s) to generate 'liking_X' traits for",
    )
    parser.add_argument(
        "--trait", type=str, help="Custom trait name (e.g., 'liking_owls')"
    )
    parser.add_argument("--description", type=str, help="Custom trait description")
    parser.add_argument(
        "--question_instruction",
        type=str,
        default="",
        help="Additional instructions for question generation",
    )

    args = parser.parse_args()

    base_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    if args.animal:
        for animal in args.animal:
            animal = animal.lower()
            trait = f"liking_{_pluralize(animal)}"
            description = _animal_description(animal)

            try:
                trait_data = generate_trait_data(
                    trait, description, args.question_instruction
                )
                save_trait_data(trait, trait_data, base_dir)
                print(f"Successfully generated trait data for '{trait}'\n")
            except Exception as e:
                print(f"Error generating trait data for '{trait}': {e}\n")

    elif args.trait and args.description:
        try:
            trait_data = generate_trait_data(
                args.trait, args.description, args.question_instruction
            )
            save_trait_data(args.trait, trait_data, base_dir)
            print(f"Successfully generated trait data for '{args.trait}'")
        except Exception as e:
            print(f"Error generating trait data for '{args.trait}': {e}")

    else:
        parser.print_help()
        print("\nExamples:")
        print(
            "  python src/data_generation/generate_trait_data.py --animal eagle lion phoenix"
        )


if __name__ == "__main__":
    main()
