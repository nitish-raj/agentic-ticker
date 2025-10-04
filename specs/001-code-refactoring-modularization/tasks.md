# Tasks: Code Refactoring and Modularization

**Input**: Design documents from `/specs/001-code-refactoring-modularization/`
**Prerequisites**: plan.md (required), research.md, data-model.md, contracts/

## Execution Flow (main)
```
1. Load plan.md from feature directory
   → If not found: ERROR "No implementation plan found"
   → Extract: tech stack, libraries, structure
2. Load optional design documents:
   → data-model.md: Extract entities → model tasks
   → contracts/: Each file → contract test task
   → research.md: Extract decisions → setup tasks
3. Generate tasks by category:
   → Setup: project init, dependencies, linting
   → Tests: contract tests, integration tests
   → Core: models, services, CLI commands
   → Integration: DB, middleware, logging
   → Polish: unit tests, performance, docs
4. Apply task rules:
   → Different files = mark [P] for parallel
   → Same file = sequential (no [P])
   → Tests before implementation (TDD)
5. Number tasks sequentially (T001, T002...)
6. Generate dependency graph
7. Create parallel execution examples
8. Validate task completeness:
   → All contracts have tests?
   → All entities have models?
   → All endpoints implemented?
9. Return: SUCCESS (tasks ready for execution)
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Path Conventions
- **Single project**: `src/`, `tests/` at repository root
- **Web app**: `backend/src/`, `frontend/src/`
- **Mobile**: `api/src/`, `ios/src/` or `android/src/`
- Paths shown below assume single project - adjust based on plan.md structure

## Phase 3.1: Setup
- [X] T001 Verify environment setup and dependencies in requirements.txt
- [X] T002 Run baseline tests and linting to establish current state
- [X] T003 [P] Create test directories for contract, integration, and unit tests

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3
**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**
- [X] T004 [P] Contract test POST /utility-modules in tests/contract/test_utility_modules_post.py
- [X] T005 [P] Contract test GET /utility-modules/{moduleName} in tests/contract/test_utility_modules_get.py
- [X] T006 [P] Contract test PUT /utility-modules/{moduleName} in tests/contract/test_utility_modules_put.py
- [X] T007 [P] Contract test DELETE /utility-modules/{moduleName} in tests/contract/test_utility_modules_delete.py
- [X] T008 [P] Contract test POST /decorators in tests/contract/test_decorators_post.py
- [X] T009 [P] Contract test GET /code-duplication-patterns in tests/contract/test_code_duplication_patterns_get.py
- [X] T010 [P] Contract test POST /code-duplication-patterns in tests/contract/test_code_duplication_patterns_post.py
- [X] T011 [P] Contract test GET /refactoring-progress in tests/contract/test_refactoring_progress_get.py
- [X] T012 [P] Integration test utility module creation workflow in tests/integration/test_utility_module_creation.py
- [X] T013 [P] Integration test decorator application workflow in tests/integration/test_decorator_application.py
- [X] T014 [P] Integration test code duplication elimination workflow in tests/integration/test_duplication_elimination.py

## Phase 3.3: Core Implementation (ONLY after tests are failing)
- [X] T015 [P] UtilityModule entity implementation in src/models/utility_module.py
- [X] T016 [P] UtilityFunction entity implementation in src/models/utility_function.py
- [X] T017 [P] Decorator entity implementation in src/models/decorator.py
- [X] T018 [P] CodeDuplicationPattern entity implementation in src/models/code_duplication_pattern.py
- [X] T019 [P] CodeLocation entity implementation in src/models/code_location.py
- [X] T020 [P] RefactoringProgress entity implementation in src/models/refactoring_progress.py
- [X] T021 [P] Create gemini_api.py utility module in src/gemini_api.py
- [X] T022 [P] Create search_utils.py utility module in src/search_utils.py
- [X] T023 [P] Create date_utils.py utility module in src/date_utils.py
- [X] T024 [P] Create decorators.py utility module in src/decorators.py
- [X] T025 [P] Create validation_utils.py utility module in src/validation_utils.py
- [X] T026 [P] Create chart_utils.py utility module in src/chart_utils.py
- [X] T027 [P] Create config.py utility module in src/config.py
- [X] T028 [P] Create json_helpers_utils.py utility module in src/json_helpers_utils.py
- [X] T029 Update services.py to use new utility modules and remove duplication
- [X] T030 Update orchestrator.py to use new decorators and utility modules
- [X] T031 Update ui_components.py to use new utility modules and remove duplication
- [X] T032 Update agentic_ticker.py to use new utility modules and remove duplication

## Phase 3.4: Integration
- [X] T033 Integrate utility modules with existing error handling patterns
- [X] T034 Connect decorators to cross-cutting concerns across all modules
- [X] T035 Implement configuration management integration across all utilities
- [X] T036 Add logging and monitoring integration for all new utilities
- [X] T037 Implement backward compatibility layer for existing function calls
- [X] T038 Create utility module registry for dynamic module loading
- [X] T039 Implement code duplication detection and tracking system
- [X] T040 Create refactoring progress tracking and reporting system

## Phase 3.5: Polish
- [X] T041 [P] Unit tests for gemini_api.py in tests/unit/test_gemini_api.py
- [X] T042 [P] Unit tests for search_utils.py in tests/unit/test_search_utils.py
- [X] T043 [P] Unit tests for date_utils.py in tests/unit/test_date_utils.py
- [X] T044 [P] Unit tests for decorators.py in tests/unit/test_decorators.py
- [X] T045 [P] Unit tests for validation_utils.py in tests/unit/test_validation_utils.py
- [X] T046 [P] Unit tests for chart_utils.py in tests/unit/test_chart_utils.py
- [X] T047 [P] Unit tests for config.py in tests/unit/test_config.py
- [X] T048 [P] Unit tests for json_helpers_utils.py in tests/unit/test_json_helpers_utils.py
- [X] T049 [P] Unit tests for all entity models in tests/unit/test_models.py
- [X] T050 Performance tests for API operations (<200ms p95) in tests/performance/test_api_performance.py
- [X] T051 [P] Update documentation for all new utility modules
- [X] T052 [P] Update AGENTS.md with new utility technologies and patterns
- [X] T053 [P] Update README.md with refactoring changes and improvements
- [X] T054 Run comprehensive integration tests to validate all functionality
- [X] T055 Verify code reduction metrics (35-40% reduction target)
- [X] T056 Final validation using quickstart.md guide

## Dependencies
- Tests (T004-T014) before implementation (T015-T032)
- Entity models (T015-T020) before utility modules (T021-T028)
- Utility modules (T021-T028) before integration (T033-T040)
- Integration (T033-T040) before polish (T041-T056)
- T029-T032 (updating existing files) must be sequential as they modify the same files
- T041-T049 (unit tests) can run in parallel as they test different modules
- T051-T053 (documentation updates) can run in parallel

## Parallel Example
```
# Launch T004-T011 together (contract tests):
Task: "Contract test POST /utility-modules in tests/contract/test_utility_modules_post.py"
Task: "Contract test GET /utility-modules/{moduleName} in tests/contract/test_utility_modules_get.py"
Task: "Contract test PUT /utility-modules/{moduleName} in tests/contract/test_utility_modules_put.py"
Task: "Contract test DELETE /utility-modules/{moduleName} in tests/contract/test_utility_modules_delete.py"
Task: "Contract test POST /decorators in tests/contract/test_decorators_post.py"
Task: "Contract test GET /code-duplication-patterns in tests/contract/test_code_duplication_patterns_get.py"
Task: "Contract test POST /code-duplication-patterns in tests/contract/test_code_duplication_patterns_post.py"
Task: "Contract test GET /refactoring-progress in tests/contract/test_refactoring_progress_get.py"

# Launch T012-T014 together (integration tests):
Task: "Integration test utility module creation workflow in tests/integration/test_utility_module_creation.py"
Task: "Integration test decorator application workflow in tests/integration/test_decorator_application.py"
Task: "Integration test code duplication elimination workflow in tests/integration/test_duplication_elimination.py"

# Launch T015-T020 together (entity models):
Task: "UtilityModule entity implementation in src/models/utility_module.py"
Task: "UtilityFunction entity implementation in src/models/utility_function.py"
Task: "Decorator entity implementation in src/models/decorator.py"
Task: "CodeDuplicationPattern entity implementation in src/models/code_duplication_pattern.py"
Task: "CodeLocation entity implementation in src/models/code_location.py"
Task: "RefactoringProgress entity implementation in src/models/refactoring_progress.py"

# Launch T021-T028 together (utility modules):
Task: "Create gemini_api.py utility module in src/gemini_api.py"
Task: "Create search_utils.py utility module in src/search_utils.py"
Task: "Create date_utils.py utility module in src/date_utils.py"
Task: "Create decorators.py utility module in src/decorators.py"
Task: "Create validation_utils.py utility module in src/validation_utils.py"
Task: "Create chart_utils.py utility module in src/chart_utils.py"
Task: "Create config.py utility module in src/config.py"
Task: "Create json_helpers_utils.py utility module in src/json_helpers_utils.py"

# Launch T041-T049 together (unit tests):
Task: "Unit tests for gemini_api.py in tests/unit/test_gemini_api.py"
Task: "Unit tests for search_utils.py in tests/unit/test_search_utils.py"
Task: "Unit tests for date_utils.py in tests/unit/test_date_utils.py"
Task: "Unit tests for decorators.py in tests/unit/test_decorators.py"
Task: "Unit tests for validation_utils.py in tests/unit/test_validation_utils.py"
Task: "Unit tests for chart_utils.py in tests/unit/test_chart_utils.py"
Task: "Unit tests for config.py in tests/unit/test_config.py"
Task: "Unit tests for json_helpers_utils.py in tests/unit/test_json_helpers_utils.py"
Task: "Unit tests for all entity models in tests/unit/test_models.py"

# Launch T051-T053 together (documentation):
Task: "Update documentation for all new utility modules"
Task: "Update AGENTS.md with new utility technologies and patterns"
Task: "Update README.md with refactoring changes and improvements"
```

## Notes
- [P] tasks = different files, no dependencies
- Verify tests fail before implementing
- Commit after each task
- Avoid: vague tasks, same file conflicts
- T029-T032 must be sequential as they modify existing source files
- Focus on high-priority refactoring items first (gemini_api, search_utils, date_utils, decorators)

## Task Generation Rules
*Applied during main() execution*

1. **From Contracts**:
   - Each contract file → contract test task [P]
   - Each endpoint → implementation task
   
2. **From Data Model**:
   - Each entity → model creation task [P]
   - Relationships → service layer tasks
   
3. **From User Stories**:
   - Each story → integration test [P]
   - Quickstart scenarios → validation tasks

4. **Ordering**:
   - Setup → Tests → Models → Services → Endpoints → Polish
   - Dependencies block parallel execution

## Validation Checklist
*GATE: Checked by main() before returning*

- [x] All contracts have corresponding tests
- [x] All entities have model tasks
- [x] All tests come before implementation
- [x] Parallel tasks truly independent
- [x] Each task specifies exact file path
- [x] No task modifies same file as another [P] task (except T029-T032 which are intentionally sequential)