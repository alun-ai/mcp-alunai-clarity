#!/usr/bin/env python3
"""
Real User Scenarios for Enhanced Structured Thinking System

These scenarios simulate actual user interactions and workflows to validate
that the enhanced structured thinking system works effectively in practice.
"""

from tests.integration.test_structured_thinking_real_users import TestScenario


def get_real_user_scenarios():
    """Get all real user test scenarios organized by feature category."""
    
    scenarios = []
    
    # ============================================================================
    # 1. AUTO-PROGRESSION SCENARIOS
    # ============================================================================
    
    scenarios.append(TestScenario(
        name="auto_progression_web_app_development",
        description="User asks for help building a web app, system should auto-progress through thinking stages",
        user_input="I need to build a modern web application for managing customer orders with real-time updates, user authentication, and payment processing.",
        context={
            "project_context": {
                "language": "javascript", 
                "framework": "react",
                "detected_frameworks": ["react", "nodejs"],
                "complexity": 0.8
            },
            "files_accessed": [],
            "commands_executed": [],
            "recent_activity": ["Starting new project", "Setting up development environment"]
        },
        expected_outcomes=[
            "thinking_session_created",
            "auto_progression_successful", 
            "multi_stage_progression_2",
            "action_plan_generated"
        ],
        validation_criteria={
            "max_execution_time": 30.0,
            "min_confidence": 0.7,
            "expected_components": ["user authentication", "payment processing", "real-time updates", "database"]
        },
        difficulty="intermediate",
        estimated_time_minutes=20
    ))
    
    scenarios.append(TestScenario(
        name="auto_progression_microservices_architecture",
        description="Complex architectural problem should trigger comprehensive auto-progression",
        user_input="Design a microservices architecture for an e-commerce platform that handles 100k users, includes inventory management, order processing, user profiles, and recommendation engine with machine learning.",
        context={
            "project_context": {
                "language": "python",
                "framework": "fastapi", 
                "platform": "aws",
                "detected_frameworks": ["fastapi", "docker", "kubernetes"],
                "complexity": 0.95
            },
            "files_accessed": ["/architecture/services.yaml", "/docker/compose.yml"],
            "commands_executed": ["kubectl get pods", "docker ps"],
            "recent_activity": ["Researching microservices patterns", "Planning service boundaries"]
        },
        expected_outcomes=[
            "thinking_session_created",
            "auto_progression_successful",
            "multi_stage_progression_2", 
            "multi_stage_progression_3",
            "action_plan_generated"
        ],
        validation_criteria={
            "max_execution_time": 45.0,
            "min_confidence": 0.8,
            "min_components": 8,
            "expected_components": ["microservices", "kubernetes", "machine learning", "recommendation engine"]
        },
        difficulty="advanced",
        estimated_time_minutes=30
    ))
    
    scenarios.append(TestScenario(
        name="auto_progression_api_integration",
        description="Simple API integration task should have measured auto-progression",
        user_input="Help me integrate with the Stripe payment API for processing credit card payments in my Node.js application.",
        context={
            "project_context": {
                "language": "javascript",
                "framework": "express",
                "detected_frameworks": ["express", "nodejs"]
            },
            "files_accessed": ["/routes/payments.js", "/package.json"],
            "commands_executed": ["npm install stripe"],
            "recent_activity": ["Reading Stripe documentation"]
        },
        expected_outcomes=[
            "thinking_session_created",
            "auto_progression_successful",
            "action_plan_generated"
        ],
        validation_criteria={
            "max_execution_time": 20.0,
            "min_confidence": 0.6,
            "expected_components": ["api integration", "payment processing", "error handling"]
        },
        difficulty="beginner",
        estimated_time_minutes=15
    ))
    
    # ============================================================================
    # 2. PROACTIVE SUGGESTIONS SCENARIOS  
    # ============================================================================
    
    scenarios.append(TestScenario(
        name="proactive_suggestions_debugging_complex_issue",
        description="Multiple error patterns should trigger debugging strategy suggestions",
        user_input="I'm having performance issues in my distributed system and users are reporting timeouts",
        context={
            "current_task": "Debug distributed system performance issues",
            "project_context": {
                "detected_frameworks": ["kubernetes", "microservices", "redis"],
                "languages": ["python", "go"]
            },
            "files_accessed": [
                "/logs/service1.log", "/logs/service2.log", "/logs/gateway.log",
                "/monitoring/metrics.json", "/config/kubernetes.yaml"
            ],
            "commands_executed": [
                "kubectl logs service1 | grep ERROR",
                "docker stats", 
                "curl -X GET /health/service2",
                "grep 'timeout' /logs/*.log"
            ],
            "recent_activity": [
                "Investigating timeout errors",
                "Checking service health endpoints", 
                "Analyzing resource usage"
            ]
        },
        expected_outcomes=[
            "suggestions_generated",
            "high_confidence_suggestions_available",
            "auto_trigger_successful",
            "enhanced_suggestions_generated"
        ],
        validation_criteria={
            "min_suggestions": 3,
            "min_confidence": 0.8,
            "expected_components": ["debugging strategy", "performance optimization", "monitoring"]
        },
        difficulty="advanced",
        estimated_time_minutes=25
    ))
    
    scenarios.append(TestScenario(
        name="proactive_suggestions_new_technology_learning",
        description="Working with new technologies should trigger learning consolidation suggestions",
        user_input="I'm learning Docker and Kubernetes for container orchestration",
        context={
            "current_task": "Learn container orchestration with Kubernetes",
            "project_context": {
                "detected_frameworks": ["docker", "kubernetes"],
                "languages": ["yaml"]
            },
            "files_accessed": [
                "/k8s/deployment.yaml", "/k8s/service.yaml", "/Dockerfile",
                "/docs/kubernetes-tutorial.md"
            ],
            "commands_executed": [
                "docker build -t myapp .",
                "kubectl apply -f deployment.yaml",
                "kubectl get pods",
                "docker run -p 8080:8080 myapp"
            ],
            "recent_activity": [
                "Reading Kubernetes documentation",
                "Creating first deployment",
                "Learning about services and ingress"
            ]
        },
        expected_outcomes=[
            "suggestions_generated",
            "high_confidence_suggestions_available",
            "enhanced_suggestions_generated",
            "session_tracking_active"
        ],
        validation_criteria={
            "min_suggestions": 2,
            "min_confidence": 0.7,
            "expected_components": ["learning consolidation", "docker", "kubernetes"]
        },
        difficulty="intermediate",
        estimated_time_minutes=18
    ))
    
    scenarios.append(TestScenario(
        name="proactive_suggestions_architecture_decisions",
        description="Multiple architectural choices should trigger decision analysis suggestions",
        user_input="Should I use GraphQL or REST API for my mobile app backend? Also considering MongoDB vs PostgreSQL for data storage.",
        context={
            "current_task": "Choose architecture for mobile app backend",
            "project_context": {
                "detected_frameworks": ["mobile", "backend"],
                "languages": ["javascript", "typescript"]
            },
            "files_accessed": ["/research/graphql-vs-rest.md", "/research/database-comparison.md"],
            "commands_executed": ["npm init", "npm search graphql"],
            "recent_activity": [
                "Researching GraphQL benefits",
                "Comparing database options",
                "Reading architecture articles",
                "Discussing with team about trade-offs"
            ]
        },
        expected_outcomes=[
            "suggestions_generated",
            "high_confidence_suggestions_available",
            "auto_trigger_successful"
        ],
        validation_criteria={
            "min_suggestions": 1,
            "min_confidence": 0.8,
            "expected_components": ["decision analysis", "architecture", "database", "api design"]
        },
        difficulty="intermediate",
        estimated_time_minutes=20
    ))
    
    # ============================================================================
    # 3. COMPONENT DETECTION SCENARIOS
    # ============================================================================
    
    scenarios.append(TestScenario(
        name="component_detection_ecommerce_platform",
        description="Complex e-commerce description should detect many architectural components",
        user_input="Build a comprehensive e-commerce platform with user registration, product catalog with search and filters, shopping cart, checkout with multiple payment options including PayPal and Stripe, order tracking, inventory management, admin dashboard, customer support chat, email notifications, mobile app API, and analytics dashboard.",
        context={
            "project_context": {
                "language": "python",
                "framework": "django",
                "platform": "aws",
                "technologies": ["postgresql", "redis", "celery", "docker"]
            }
        },
        expected_outcomes=[
            "components_detected",
            "high_confidence_detection", 
            "risk_factors_identified",
            "complexity_assessed"
        ],
        validation_criteria={
            "min_components": 10,
            "min_confidence": 0.8,
            "expected_components": [
                "user management", "payment processing", "database", "api design",
                "authentication", "real-time features", "mobile app", "analytics"
            ]
        },
        difficulty="advanced",
        estimated_time_minutes=12
    ))
    
    scenarios.append(TestScenario(
        name="component_detection_devops_pipeline",
        description="DevOps and CI/CD requirements should detect infrastructure components",
        user_input="Set up a complete DevOps pipeline with automated testing, code quality checks, security scanning, Docker containerization, Kubernetes deployment, monitoring with Prometheus and Grafana, log aggregation, and automated rollback capabilities.",
        context={
            "project_context": {
                "platform": "aws",
                "technologies": ["docker", "kubernetes", "jenkins", "prometheus"]
            }
        },
        expected_outcomes=[
            "components_detected",
            "high_confidence_detection",
            "complexity_assessed",
            "detected_continuous_integration_and_deployment",
            "detected_monitoring_logging_and_observability"
        ],
        validation_criteria={
            "min_components": 8,
            "min_confidence": 0.7,
            "expected_components": [
                "ci/cd", "docker", "kubernetes", "monitoring", "security", "testing"
            ]
        },
        difficulty="advanced", 
        estimated_time_minutes=10
    ))
    
    scenarios.append(TestScenario(
        name="component_detection_simple_crud_app",
        description="Simple CRUD application should detect basic components accurately",
        user_input="Create a simple blog application where users can create, read, update, and delete blog posts with basic authentication.",
        context={
            "project_context": {
                "language": "python",
                "framework": "flask"
            }
        },
        expected_outcomes=[
            "components_detected",
            "high_confidence_detection",
            "detected_user_management_and_authentication",
            "detected_core_implementation_logic"
        ],
        validation_criteria={
            "min_components": 4,
            "min_confidence": 0.7,
            "expected_components": ["authentication", "database", "crud", "web framework"]
        },
        difficulty="beginner",
        estimated_time_minutes=8
    ))
    
    # ============================================================================
    # 4. CONTEXT INTEGRATION SCENARIOS
    # ============================================================================
    
    scenarios.append(TestScenario(
        name="context_integration_multi_file_refactoring",
        description="Working across multiple files should build rich context for suggestions",
        user_input="Refactor this monolithic application into microservices architecture",
        context={
            "current_task": "Refactor monolith to microservices",
            "project_context": {
                "detected_frameworks": ["django", "postgresql", "redis"],
                "detected_languages": ["python"],
                "project_complexity": 0.9,
                "architecture_patterns": ["monolithic", "mvc"]
            },
            "files_accessed": [
                "/app/models/user.py", "/app/models/order.py", "/app/models/product.py",
                "/app/views/api.py", "/app/views/admin.py", "/app/services/payment.py",
                "/app/services/notification.py", "/config/settings.py", "/requirements.txt"
            ],
            "commands_executed": [
                "python manage.py migrate", 
                "docker build -t monolith .",
                "grep -r 'class.*Model' app/models/",
                "find . -name '*.py' | wc -l"
            ],
            "recent_activity": [
                "Analyzing current architecture",
                "Identifying service boundaries", 
                "Planning data migration strategy"
            ]
        },
        expected_outcomes=[
            "meaningful_complexity_detected",
            "intelligence_classification_accurate",
            "multi_dimensional_analysis_enabled",
            "memory_integration_enabled",
            "context_driven_suggestions_successful",
            "hook_integration_working"
        ],
        validation_criteria={
            "min_confidence": 0.8,
            "expected_components": ["microservices", "architecture", "refactoring", "data migration"]
        },
        difficulty="advanced",
        estimated_time_minutes=25
    ))
    
    scenarios.append(TestScenario(
        name="context_integration_performance_optimization",
        description="Performance investigation with logs and metrics should provide rich context",
        user_input="My application is slow and I need to identify bottlenecks",
        context={
            "current_task": "Identify and fix performance bottlenecks",
            "project_context": {
                "detected_frameworks": ["nodejs", "express", "mongodb"],
                "detected_languages": ["javascript"],
                "project_complexity": 0.7
            },
            "files_accessed": [
                "/logs/application.log", "/logs/database.log", "/logs/nginx.log",
                "/monitoring/cpu_usage.json", "/monitoring/memory_usage.json",
                "/src/routes/api.js", "/src/models/user.js", "/package.json"
            ],
            "commands_executed": [
                "tail -f /logs/application.log",
                "htop", "free -m", "iostat", 
                "node --prof app.js",
                "mongostat", "mongotop"
            ],
            "recent_activity": [
                "Monitoring application performance",
                "Analyzing slow queries",
                "Checking system resources"
            ]
        },
        expected_outcomes=[
            "meaningful_complexity_detected",
            "intelligence_classification_accurate", 
            "context_driven_suggestions_successful",
            "auto_execution_available"
        ],
        validation_criteria={
            "min_confidence": 0.7,
            "expected_components": ["performance optimization", "database", "monitoring"]
        },
        difficulty="intermediate",
        estimated_time_minutes=20
    ))
    
    # ============================================================================
    # 5. END-TO-END INTEGRATION SCENARIOS
    # ============================================================================
    
    scenarios.append(TestScenario(
        name="end_to_end_startup_mvp_development",
        description="Complete startup MVP development should exercise all enhanced features",
        user_input="I'm starting a SaaS company and need to build an MVP for a project management tool with team collaboration, real-time updates, file sharing, and billing integration. I have 3 months to launch.",
        context={
            "current_task": "Build SaaS MVP for project management tool",
            "project_context": {
                "language": "typescript",
                "framework": "nextjs", 
                "platform": "vercel",
                "detected_frameworks": ["react", "nextjs", "nodejs"],
                "detected_languages": ["typescript", "javascript"],
                "project_complexity": 0.85
            },
            "files_accessed": ["/planning/requirements.md", "/research/competitors.md"],
            "commands_executed": ["npx create-next-app@latest", "npm install"],
            "recent_activity": [
                "Researching competitor features",
                "Planning MVP scope",
                "Setting up development environment",
                "Creating user stories"
            ]
        },
        expected_outcomes=[
            "initial_suggestions_generated",
            "auto_trigger_successful", 
            "end_to_end_automation_successful",
            "thinking_session_created",
            "auto_progression_successful",
            "action_plan_generated"
        ],
        validation_criteria={
            "max_execution_time": 60.0,
            "min_confidence": 0.8,
            "min_components": 8,
            "expected_components": [
                "saas", "billing", "real-time", "file sharing", "collaboration", 
                "authentication", "project management"
            ]
        },
        difficulty="advanced",
        estimated_time_minutes=35
    ))
    
    scenarios.append(TestScenario(
        name="end_to_end_legacy_system_modernization",
        description="Legacy modernization should trigger comprehensive analysis and planning",
        user_input="We have a 10-year-old PHP monolith with MySQL that needs modernization. It handles customer orders, inventory, and reporting. We want to move to microservices with modern tech stack while maintaining business continuity.",
        context={
            "current_task": "Modernize legacy PHP monolith to microservices",
            "project_context": {
                "detected_frameworks": ["php", "mysql", "apache"],
                "detected_languages": ["php", "sql"],
                "project_complexity": 0.95
            },
            "files_accessed": [
                "/legacy/orders.php", "/legacy/inventory.php", "/legacy/reports.php",
                "/database/schema.sql", "/docs/business_rules.md"
            ],
            "commands_executed": [
                "find . -name '*.php' | wc -l",
                "mysql -e 'SHOW TABLES'",
                "grep -r 'function' *.php | wc -l"
            ],
            "recent_activity": [
                "Auditing legacy codebase",
                "Documenting business logic",
                "Planning migration strategy",
                "Identifying technical debt"
            ]
        },
        expected_outcomes=[
            "initial_suggestions_generated",
            "auto_trigger_successful",
            "thinking_session_created", 
            "auto_progression_successful",
            "multi_stage_progression_2",
            "action_plan_generated",
            "end_to_end_automation_successful"
        ],
        validation_criteria={
            "max_execution_time": 90.0,
            "min_confidence": 0.7,
            "min_components": 10,
            "expected_components": [
                "legacy modernization", "microservices", "migration", "database",
                "business continuity", "technical debt"
            ]
        },
        difficulty="advanced",
        estimated_time_minutes=45
    ))
    
    # ============================================================================
    # 6. EDGE CASE AND STRESS TEST SCENARIOS
    # ============================================================================
    
    scenarios.append(TestScenario(
        name="stress_test_minimal_context",
        description="System should handle scenarios with minimal context gracefully",
        user_input="Help me code something",
        context={
            "current_task": "Code something",
            "project_context": {},
            "files_accessed": [],
            "commands_executed": [],
            "recent_activity": []
        },
        expected_outcomes=[
            "suggestions_generated"  # Should still generate some suggestions
        ],
        validation_criteria={
            "max_execution_time": 15.0,
            "min_suggestions": 1
        },
        difficulty="beginner",
        estimated_time_minutes=5
    ))
    
    scenarios.append(TestScenario(
        name="stress_test_extremely_complex_context",
        description="System should handle very complex scenarios without performance degradation",
        user_input="Design and implement a complete cloud-native platform with microservices, event-driven architecture, CQRS, event sourcing, API gateway, service mesh, observability, security, multi-tenant SaaS, global CDN, real-time analytics, machine learning pipelines, and compliance with SOC2 and GDPR.",
        context={
            "current_task": "Build enterprise cloud-native platform",
            "project_context": {
                "detected_frameworks": [
                    "kubernetes", "istio", "kafka", "elasticsearch", "prometheus",
                    "grafana", "jaeger", "consul", "vault", "terraform"
                ],
                "detected_languages": ["go", "python", "typescript", "rust"],
                "project_complexity": 1.0,
                "architecture_patterns": [
                    "microservices", "event-driven", "cqrs", "event-sourcing"
                ]
            },
            "files_accessed": [f"/services/service{i}.go" for i in range(20)] + 
                           [f"/config/k8s/deployment{i}.yaml" for i in range(15)] +
                           ["/architecture/design.md", "/security/policies.yaml"],
            "commands_executed": [
                "kubectl get pods", "terraform plan", "docker build",
                "go test ./...", "helm upgrade", "istioctl analyze"
            ] * 5,  # Simulate many commands
            "recent_activity": [
                "Designing system architecture",
                "Setting up development environment",
                "Implementing core services",
                "Configuring security policies",
                "Planning deployment strategy"
            ] * 3  # Simulate extensive activity
        },
        expected_outcomes=[
            "components_detected",
            "high_confidence_detection",
            "complexity_assessed",
            "meaningful_complexity_detected"
        ],
        validation_criteria={
            "max_execution_time": 120.0,  # Allow more time for complex scenarios
            "min_components": 15,
            "min_confidence": 0.8
        },
        difficulty="advanced",
        estimated_time_minutes=60
    ))
    
    return scenarios


def get_scenarios_by_category():
    """Get scenarios organized by category for targeted testing."""
    scenarios = get_real_user_scenarios()
    
    categories = {
        "auto_progression": [s for s in scenarios if "auto_progression" in s.name],
        "proactive_suggestions": [s for s in scenarios if "proactive_suggestions" in s.name],
        "component_detection": [s for s in scenarios if "component_detection" in s.name],
        "context_integration": [s for s in scenarios if "context_integration" in s.name],
        "end_to_end": [s for s in scenarios if "end_to_end" in s.name],
        "stress_tests": [s for s in scenarios if "stress_test" in s.name]
    }
    
    return categories


def get_scenarios_by_difficulty():
    """Get scenarios organized by difficulty level."""
    scenarios = get_real_user_scenarios()
    
    difficulties = {
        "beginner": [s for s in scenarios if s.difficulty == "beginner"],
        "intermediate": [s for s in scenarios if s.difficulty == "intermediate"], 
        "advanced": [s for s in scenarios if s.difficulty == "advanced"]
    }
    
    return difficulties