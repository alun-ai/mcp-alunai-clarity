#!/usr/bin/env python3
"""
Real User Test Suite for Enhanced Structured Thinking System

This test suite simulates real user scenarios to validate that the enhanced 
structured thinking system works as intended in practical situations.
"""

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from loguru import logger

from clarity.autocode.hook_manager import HookManager
from clarity.autocode.hooks import AutoCodeHooks
from clarity.domains.memory import MemoryDomain
from clarity.mcp.server import MemoryMCPServer
from clarity.shared.config import Config


@dataclass
class TestScenario:
    """Represents a real user test scenario."""
    name: str
    description: str
    user_input: str
    context: Dict[str, Any]
    expected_outcomes: List[str]
    validation_criteria: Dict[str, Any]
    difficulty: str  # "beginner", "intermediate", "advanced"
    estimated_time_minutes: int


@dataclass 
class TestResult:
    """Results from running a test scenario."""
    scenario_name: str
    success: bool
    execution_time_seconds: float
    outcomes_achieved: List[str]
    validation_results: Dict[str, bool]
    error_messages: List[str]
    performance_metrics: Dict[str, float]
    user_experience_score: float  # 0-10 scale


class RealUserTestFramework:
    """Framework for running real user scenarios and validating results."""
    
    def __init__(self):
        self.domain_manager = None
        self.hook_manager = None
        self.mcp_server = None
        self.test_results: List[TestResult] = []
        
    async def initialize(self):
        """Initialize the test environment."""
        logger.info("Initializing real user test framework...")
        
        # Initialize domain manager
        config = Config()
        self.domain_manager = config.get_domain_manager()
        await self.domain_manager.initialize()
        
        # Initialize hooks
        autocode_hooks = AutoCodeHooks(self.domain_manager)
        self.hook_manager = HookManager(self.domain_manager, autocode_hooks)
        
        # Initialize MCP server for tool testing
        self.mcp_server = MemoryMCPServer(config.config)
        
        logger.info("Test framework initialized successfully")
    
    async def run_scenario(self, scenario: TestScenario) -> TestResult:
        """Run a single test scenario and validate results."""
        logger.info(f"Running scenario: {scenario.name}")
        start_time = time.time()
        
        result = TestResult(
            scenario_name=scenario.name,
            success=False,
            execution_time_seconds=0.0,
            outcomes_achieved=[],
            validation_results={},
            error_messages=[],
            performance_metrics={},
            user_experience_score=0.0
        )
        
        try:
            # Execute the scenario based on its type
            if "auto_progression" in scenario.name.lower():
                await self._test_auto_progression_scenario(scenario, result)
            elif "proactive_suggestions" in scenario.name.lower():
                await self._test_proactive_suggestions_scenario(scenario, result)
            elif "component_detection" in scenario.name.lower():
                await self._test_component_detection_scenario(scenario, result)
            elif "context_integration" in scenario.name.lower():
                await self._test_context_integration_scenario(scenario, result)
            else:
                await self._test_general_scenario(scenario, result)
            
            # Calculate execution time
            result.execution_time_seconds = time.time() - start_time
            
            # Validate results
            await self._validate_scenario_results(scenario, result)
            
            # Calculate user experience score
            result.user_experience_score = self._calculate_user_experience_score(scenario, result)
            
        except Exception as e:
            result.error_messages.append(str(e))
            logger.error(f"Error in scenario {scenario.name}: {e}")
        
        self.test_results.append(result)
        return result
    
    async def _test_auto_progression_scenario(self, scenario: TestScenario, result: TestResult):
        """Test automatic stage progression scenarios."""
        if not self.hook_manager.structured_thinking_extension:
            result.error_messages.append("Structured thinking extension not available")
            return
        
        # Create initial thinking session
        analysis_result = await self.hook_manager.structured_thinking_extension.analyze_problem_with_stages(
            problem=scenario.user_input,
            project_context=scenario.context.get("project_context", {})
        )
        
        if analysis_result.get("error"):
            result.error_messages.append(f"Failed to create session: {analysis_result['error']}")
            return
        
        session_id = analysis_result["session_id"]
        result.outcomes_achieved.append("thinking_session_created")
        
        # Test automatic progression
        progression_start = time.time()
        progression_result = await self.hook_manager.structured_thinking_extension.auto_progress_thinking_stage(
            session_id=session_id,
            auto_execute=True
        )
        progression_time = time.time() - progression_start
        
        result.performance_metrics["auto_progression_time"] = progression_time
        
        if progression_result.get("status") == "auto_progressed":
            result.outcomes_achieved.append("auto_progression_successful")
            result.performance_metrics["progression_confidence"] = progression_result.get("confidence", 0)
            
            # Test multiple progressions if supported
            for i in range(2):  # Try progressing 2 more stages
                next_progression = await self.hook_manager.structured_thinking_extension.auto_progress_thinking_stage(
                    session_id=session_id,
                    auto_execute=True
                )
                if next_progression.get("status") == "auto_progressed":
                    result.outcomes_achieved.append(f"multi_stage_progression_{i+2}")
                else:
                    break
        
        # Generate final summary
        try:
            summary_result = await self.hook_manager.structured_thinking_extension.generate_coding_action_plan(session_id)
            if not summary_result.get("error"):
                result.outcomes_achieved.append("action_plan_generated")
                result.performance_metrics["action_items_count"] = len(summary_result.get("action_items", []))
        except Exception as e:
            result.error_messages.append(f"Summary generation failed: {e}")
    
    async def _test_proactive_suggestions_scenario(self, scenario: TestScenario, result: TestResult):
        """Test proactive thinking suggestions scenarios."""
        if not self.hook_manager.structured_thinking_extension:
            result.error_messages.append("Structured thinking extension not available")
            return
        
        # Test proactive suggestions
        suggestions_start = time.time()
        suggestions = await self.hook_manager.structured_thinking_extension.suggest_proactive_thinking(
            context=scenario.context,
            limit=5
        )
        suggestions_time = time.time() - suggestions_start
        
        result.performance_metrics["suggestions_generation_time"] = suggestions_time
        
        if not suggestions.get("error") and suggestions.get("suggestions"):
            result.outcomes_achieved.append("suggestions_generated")
            result.performance_metrics["suggestions_count"] = len(suggestions["suggestions"])
            
            # Check suggestion quality
            high_confidence_suggestions = [
                s for s in suggestions["suggestions"] 
                if s["confidence"] >= 0.7
            ]
            result.performance_metrics["high_confidence_suggestions"] = len(high_confidence_suggestions)
            
            if high_confidence_suggestions:
                result.outcomes_achieved.append("high_confidence_suggestions_available")
                
                # Test auto-triggering based on suggestions
                auto_trigger_result = await self.hook_manager.structured_thinking_extension.auto_trigger_thinking_from_context(
                    context=scenario.context,
                    threshold=0.8
                )
                
                if auto_trigger_result.get("status") == "auto_triggered":
                    result.outcomes_achieved.append("auto_trigger_successful")
                    result.performance_metrics["trigger_confidence"] = auto_trigger_result.get("confidence", 0)
        
        # Test enhanced suggestions through hook manager
        try:
            enhanced_suggestions = await self.hook_manager.get_enhanced_thinking_suggestions(scenario.context)
            if not enhanced_suggestions.get("error"):
                result.outcomes_achieved.append("enhanced_suggestions_generated")
                if enhanced_suggestions.get("active_sessions"):
                    result.outcomes_achieved.append("session_tracking_active")
        except Exception as e:
            result.error_messages.append(f"Enhanced suggestions failed: {e}")
    
    async def _test_component_detection_scenario(self, scenario: TestScenario, result: TestResult):
        """Test enhanced problem component detection scenarios."""
        if not self.hook_manager.structured_thinking_extension:
            result.error_messages.append("Structured thinking extension not available")
            return
        
        # Test component analysis
        detection_start = time.time()
        components = self.hook_manager.structured_thinking_extension._analyze_problem_components(
            problem=scenario.user_input,
            context=scenario.context.get("project_context", {})
        )
        detection_time = time.time() - detection_start
        
        result.performance_metrics["component_detection_time"] = detection_time
        result.performance_metrics["components_detected"] = len(components["components"])
        result.performance_metrics["detection_confidence"] = components["detection_confidence"]
        result.performance_metrics["complexity_score"] = components["complexity_indicators"]["total_complexity"]
        
        if len(components["components"]) > 0:
            result.outcomes_achieved.append("components_detected")
            
        if components["detection_confidence"] > 0.7:
            result.outcomes_achieved.append("high_confidence_detection")
            
        if len(components["risk_factors"]) > 0:
            result.outcomes_achieved.append("risk_factors_identified")
            
        if components["complexity_indicators"]["complexity_level"] in ["medium", "high"]:
            result.outcomes_achieved.append("complexity_assessed")
        
        # Validate specific component types based on scenario
        expected_components = scenario.validation_criteria.get("expected_components", [])
        detected_components_text = " ".join(components["components"]).lower()
        
        for expected in expected_components:
            if expected.lower() in detected_components_text:
                result.outcomes_achieved.append(f"detected_{expected.replace(' ', '_')}")
    
    async def _test_context_integration_scenario(self, scenario: TestScenario, result: TestResult):
        """Test smart context integration scenarios."""
        # Test context building
        context_start = time.time()
        enhanced_context = await self.hook_manager._build_enhanced_context(
            scenario.user_input,
            scenario.context
        )
        context_time = time.time() - context_start
        
        result.performance_metrics["context_building_time"] = context_time
        result.performance_metrics["context_complexity_score"] = enhanced_context["complexity_score"]
        
        if enhanced_context["complexity_score"] > 0.5:
            result.outcomes_achieved.append("meaningful_complexity_detected")
            
        if enhanced_context["intelligence_level"] in ["medium", "high"]:
            result.outcomes_achieved.append("intelligence_classification_accurate")
            
        if enhanced_context.get("multi_dimensional_analysis"):
            result.outcomes_achieved.append("multi_dimensional_analysis_enabled")
            
        if enhanced_context.get("proactive_memory_integration"):
            result.outcomes_achieved.append("memory_integration_enabled")
        
        # Test context-driven suggestions
        context_suggestions = await self.hook_manager.get_enhanced_thinking_suggestions(enhanced_context)
        if not context_suggestions.get("error"):
            result.outcomes_achieved.append("context_driven_suggestions_successful")
            
            # Check for hook integration features
            suggestions = context_suggestions.get("suggestions", [])
            hook_integrated = sum(1 for s in suggestions if s.get("hook_integration"))
            auto_executable = sum(1 for s in suggestions if s.get("auto_executable"))
            
            result.performance_metrics["hook_integrated_suggestions"] = hook_integrated
            result.performance_metrics["auto_executable_suggestions"] = auto_executable
            
            if hook_integrated > 0:
                result.outcomes_achieved.append("hook_integration_working")
            if auto_executable > 0:
                result.outcomes_achieved.append("auto_execution_available")
    
    async def _test_general_scenario(self, scenario: TestScenario, result: TestResult):
        """Test general scenarios that combine multiple features."""
        # Run a comprehensive test that exercises multiple features
        try:
            # Phase 1: Context analysis and suggestions
            suggestions = await self.hook_manager.structured_thinking_extension.suggest_proactive_thinking(
                context=scenario.context,
                limit=3
            )
            
            if suggestions.get("suggestions"):
                result.outcomes_achieved.append("initial_suggestions_generated")
                
                # Phase 2: Auto-trigger if high confidence
                high_conf_suggestions = [s for s in suggestions["suggestions"] if s["confidence"] >= 0.8]
                if high_conf_suggestions:
                    trigger_result = await self.hook_manager.structured_thinking_extension.auto_trigger_thinking_from_context(
                        context=scenario.context,
                        threshold=0.8
                    )
                    
                    if trigger_result.get("status") == "auto_triggered":
                        result.outcomes_achieved.append("auto_trigger_successful")
                        session_id = trigger_result["session_id"]
                        
                        # Phase 3: Auto-progression
                        progression_result = await self.hook_manager.structured_thinking_extension.auto_progress_thinking_stage(
                            session_id=session_id,
                            auto_execute=True
                        )
                        
                        if progression_result.get("status") == "auto_progressed":
                            result.outcomes_achieved.append("end_to_end_automation_successful")
            
        except Exception as e:
            result.error_messages.append(f"General scenario failed: {e}")
    
    async def _validate_scenario_results(self, scenario: TestScenario, result: TestResult):
        """Validate that the scenario achieved its expected outcomes."""
        for expected_outcome in scenario.expected_outcomes:
            result.validation_results[expected_outcome] = expected_outcome in result.outcomes_achieved
        
        # Check validation criteria
        criteria = scenario.validation_criteria
        
        if "min_execution_time" in criteria:
            result.validation_results["min_execution_time"] = result.execution_time_seconds >= criteria["min_execution_time"]
        
        if "max_execution_time" in criteria:
            result.validation_results["max_execution_time"] = result.execution_time_seconds <= criteria["max_execution_time"]
        
        if "min_suggestions" in criteria:
            suggestions_count = result.performance_metrics.get("suggestions_count", 0)
            result.validation_results["min_suggestions"] = suggestions_count >= criteria["min_suggestions"]
        
        if "min_confidence" in criteria:
            max_confidence = max(
                result.performance_metrics.get("progression_confidence", 0),
                result.performance_metrics.get("trigger_confidence", 0),
                0
            )
            result.validation_results["min_confidence"] = max_confidence >= criteria["min_confidence"]
        
        if "min_components" in criteria:
            components_count = result.performance_metrics.get("components_detected", 0)
            result.validation_results["min_components"] = components_count >= criteria["min_components"]
        
        # Overall success is based on achieving expected outcomes and meeting criteria
        expected_achieved = sum(1 for outcome in scenario.expected_outcomes if outcome in result.outcomes_achieved)
        expected_total = len(scenario.expected_outcomes)
        criteria_met = sum(1 for met in result.validation_results.values() if met)
        criteria_total = len(result.validation_results)
        
        success_rate = (expected_achieved + criteria_met) / (expected_total + criteria_total) if (expected_total + criteria_total) > 0 else 0
        result.success = success_rate >= 0.8  # 80% success threshold
    
    def _calculate_user_experience_score(self, scenario: TestScenario, result: TestResult) -> float:
        """Calculate a user experience score (0-10) based on various factors."""
        score = 0.0
        
        # Performance factor (0-3 points)
        if result.execution_time_seconds <= 5.0:
            score += 3.0
        elif result.execution_time_seconds <= 15.0:
            score += 2.0
        elif result.execution_time_seconds <= 30.0:
            score += 1.0
        
        # Reliability factor (0-3 points)
        if not result.error_messages:
            score += 3.0
        elif len(result.error_messages) <= 2:
            score += 1.5
        
        # Feature completeness (0-2 points)
        expected_achieved = sum(1 for outcome in scenario.expected_outcomes if outcome in result.outcomes_achieved)
        expected_total = len(scenario.expected_outcomes)
        completeness_ratio = expected_achieved / expected_total if expected_total > 0 else 0
        score += completeness_ratio * 2.0
        
        # Intelligence factor (0-2 points)
        confidence_scores = [
            result.performance_metrics.get("progression_confidence", 0),
            result.performance_metrics.get("trigger_confidence", 0),
            result.performance_metrics.get("detection_confidence", 0)
        ]
        avg_confidence = sum(s for s in confidence_scores if s > 0) / len([s for s in confidence_scores if s > 0]) if any(confidence_scores) else 0
        score += avg_confidence * 2.0
        
        return min(10.0, score)
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive test report."""
        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results if r.success)
        
        report = {
            "test_summary": {
                "total_tests_run": total_tests,
                "successful_tests": successful_tests,
                "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
                "average_execution_time": sum(r.execution_time_seconds for r in self.test_results) / total_tests if total_tests > 0 else 0,
                "average_user_experience_score": sum(r.user_experience_score for r in self.test_results) / total_tests if total_tests > 0 else 0
            },
            "performance_metrics": {
                "fastest_test": min(self.test_results, key=lambda r: r.execution_time_seconds).scenario_name if self.test_results else None,
                "slowest_test": max(self.test_results, key=lambda r: r.execution_time_seconds).scenario_name if self.test_results else None,
                "highest_ux_score": max(self.test_results, key=lambda r: r.user_experience_score).scenario_name if self.test_results else None,
                "most_errors": max(self.test_results, key=lambda r: len(r.error_messages)).scenario_name if self.test_results else None
            },
            "feature_validation": {},
            "detailed_results": [
                {
                    "scenario": r.scenario_name,
                    "success": r.success,
                    "execution_time": r.execution_time_seconds,
                    "ux_score": r.user_experience_score,
                    "outcomes_achieved": r.outcomes_achieved,
                    "error_count": len(r.error_messages)
                }
                for r in self.test_results
            ]
        }
        
        # Feature-specific validation
        features = ["auto_progression", "proactive_suggestions", "component_detection", "context_integration"]
        for feature in features:
            feature_tests = [r for r in self.test_results if feature in r.scenario_name.lower()]
            if feature_tests:
                feature_success = sum(1 for r in feature_tests if r.success) / len(feature_tests)
                report["feature_validation"][feature] = {
                    "success_rate": feature_success,
                    "test_count": len(feature_tests),
                    "average_ux_score": sum(r.user_experience_score for r in feature_tests) / len(feature_tests)
                }
        
        return report


# Test scenario definitions will be in a separate file
def get_real_user_scenarios() -> List[TestScenario]:
    """Get all real user test scenarios."""
    # This will be imported from scenario definitions
    return []


if __name__ == "__main__":
    async def main():
        framework = RealUserTestFramework()
        await framework.initialize()
        
        scenarios = get_real_user_scenarios()
        logger.info(f"Running {len(scenarios)} real user scenarios...")
        
        for scenario in scenarios:
            result = await framework.run_scenario(scenario)
            logger.info(f"Scenario '{scenario.name}': {'✓ PASS' if result.success else '✗ FAIL'}")
        
        report = framework.generate_report()
        logger.info(f"Overall success rate: {report['test_summary']['success_rate']:.2%}")
        
        return 0 if report['test_summary']['success_rate'] >= 0.8 else 1
    
    import sys
    sys.exit(asyncio.run(main()))