# AGENTS.md

## Project intent

This is a learning-oriented RAG (Retrieval-Augmented Generation) project.

Its first goal is to help developers understand the core principles of RAG and how the modules relate to each other. Implementations should therefore favour straightforward, simple, easy-to-explain solutions over complex architecture or advanced abstractions.

The rules in this file apply to the whole repository.

## Development principles

1. Prefer simple, direct, easy-to-follow code.
2. The first version does not use LangChain; do not introduce LangChain unless the user explicitly allows it.
3. Implement only the module currently asked for; do not build later modules or extra features ahead of time.
4. Every new feature needs matching tests. Documentation-only changes do not require new tests.
5. After changing code, run the relevant tests; run the full test suite when circumstances allow.
6. Do not refactor, rename, or tidy other modules without the user's permission.
7. API keys, passwords, and other secrets must never be written into the code or committed to the repository. Read them from environment variables and update `.env.example` when needed, with placeholder values only.
8. Core logic must stay clear and readable. Prefer explicit variable names, small single-purpose functions, and comments that are necessary and concise.
9. Prioritise helping the user understand the principles. When plain code expresses something clearly, do not reach for advanced wrappers, frameworks, or magic abstractions that hide the key steps.

## Scope of work

- Before changing anything, check the current module, its existing implementation, and its tests.
- Only touch the files required to finish the current task.
- Preserve existing changes in the repository that are unrelated to the current task.
- Widening the task, adding a significant dependency, or refactoring other modules all require the user's agreement first.

## Testing requirements

- Every new feature needs at least one test covering the main happy path, plus key error paths or edge cases where they apply.
- When fixing a defect, add a regression test that reproduces it whenever possible.
- After finishing the code, the relevant tests must actually be run — never just claim the tests "should pass".
- If tests cannot run because of the environment, dependencies, or an external service, state clearly which tests did not run, why, and what verification was done instead. Never claim the tests passed.

## Definition of done

A task is finished only when all of the following hold:

1. The code runs.
2. The relevant tests have been run and pass.
3. The user has been told clearly what changed, what was tested, and what the test results were.
