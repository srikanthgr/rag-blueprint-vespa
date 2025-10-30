import sys
import os
from pathlib import Path

# Add the project root to Python path so we can import from 'app' module
project_root = Path(__file__).parent.parent  # Go up from app/main.py to project root
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Set Docker socket path for macOS
os.environ['DOCKER_HOST'] = 'unix://' + os.path.expanduser('~/.docker/run/docker.sock')


from app.services import create_services
from app.profiles import (
    create_base_features_profile,
    create_collect_training_data_profile,
    create_collect_second_phase_profile,
    create_learned_linear_profile,
    create_second_with_gbdt_profile,
)
from app.schemas import create_docs_schema
from vespa.package import ApplicationPackage
from vespa.deployment import VespaDocker

if __name__ == "__main__":
    # Create rank profiles (excluding GBDT for now as it requires model file setup)
    rank_profiles = [
        create_base_features_profile(),
        create_collect_training_data_profile(),
        create_collect_second_phase_profile(),
        create_learned_linear_profile(),
        # create_second_with_gbdt_profile(),  # Commented out - requires LightGBM model file
    ]
    
    # Create schema with rank profiles
    schema = create_docs_schema()
    schema.rank_profiles = {profile.name: profile for profile in rank_profiles}
    
    # Create application package
    application_package = ApplicationPackage(
        name="rag",
        schema=[schema],  # schema parameter expects a list
        services_config=create_services(),
    )
    
    # Deploy to Vespa Docker (with increased timeout for first-time setup)
    vespa_docker = VespaDocker()
    app = vespa_docker.deploy(
        application_package=application_package,
        max_wait_configserver=300  # Wait up to 5 minutes for config server
    )