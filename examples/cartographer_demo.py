import asyncio
import os
from pathlib import Path

from agentic_patterns.domain_exploration import ExplorationBoundary
from agentic_patterns.domain_exploration import KnowledgeStore
from agentic_patterns.domain_exploration import explore_domain


async def main() -> None:
    """
    Demo of the Cartographer pattern mapping the agentic-patterns repository.
    """
    print("=== The Cartographer: Domain Exploration Demo ===")

    # 1. Configuration
    # We focus on a small subset: the first pattern (Prompt Chaining) and base models.
    boundary = ExplorationBoundary(
        max_depth=1,
        max_files=10,
        dry_run=True,
        include_patterns=["prompt_chaining.py", "_models.py", "routing.py"],
    )

    storage_path = "simple_pattern_map.json"
    root_path = "src/agentic_patterns"

    print(f"Target Folder: {root_path}")
    print(f"Target Files:  {', '.join(boundary.include_patterns)}")
    mode_str = (
        "Dry Run (AST only)" if boundary.dry_run else "Full (AST + LLM)"
    )
    print(f"Mode: {mode_str}")

    # 2. Execute Exploration
    knowledge_map = await explore_domain(
        root_path=root_path, boundary=boundary, storage_path=storage_path
    )

    # 3. Analyze Results
    store = KnowledgeStore(knowledge_map)

    print("\n=== Exploration Complete ===")
    print(f"Entities found: {len(knowledge_map.entities)}")
    print(f"Links found:    {len(knowledge_map.links)}")

    # 4. Interpret Results (Discovery Insights)
    print("\n=== Discovery Insights ===")
    central = store.find_central_entities(3)
    if central:
        print(
            f"• Found a core hub around '{central[0].name}'. "
            "This is your primary entry point."
        )
        if any(e.name == "_models.py" for e in central):
            print(
                "• Warning: '_models.py' is a central dependency. "
                "Changes here will ripple."
            )

    orphans = store.find_orphans()
    if orphans:
        print(
            f"• Detected {len(orphans)} isolated components "
            "(potential utilities or entry scripts)."
        )

    # Show Central Entities (most connected)
    print("\n--- Top Components (By Connectivity) ---")
    for i, entity in enumerate(central, 1):
        print(f"{i}. {entity.name} ({entity.entity_type})")

    print(f"\nMap saved to: {os.path.abspath(storage_path)}")
    print("\nTip: Run with boundary.dry_run=False to see semantic analysis!")


if __name__ == "__main__":
    asyncio.run(main())