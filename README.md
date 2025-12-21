https://www.kaggle.com/learn-guide/5-day-agents

**Commands**
KaggleGenAI > adk create {agent name} --model {model name} --api_key {api key}
Example: adk create research-agent --model gemini-2.5-flash-lite --api_key $GOOGLE_API_KEY

KaggleGenAI > adk run {agent_dir_name}

KaggleGenAI > adk web --port 8000

KaggleGenAI > adk web --log_level DEBUG
KaggleGenAI > adk web --log_level DEBUG --url_prefix {url_prefix}

agent_dir > python agent.py

KaggleGenAI > adk eval {agent name} {path to test case json} --config_file_path={path to test config json} --print_detailed_results
Example: KaggleGenAI > adk eval home_automation_agent home_automation_agent/integration.evalset.json --config_file_path=home_automation_agent/test_config.json --print_detailed_results

**Documentation**
https://google.github.io/adk-docs/api-reference/python/index.html
