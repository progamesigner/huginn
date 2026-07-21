# NOTE: pytest only discovers fixtures outside conftest modules when registered as plugins.
pytest_plugins = ['tests.fixtures.database']
