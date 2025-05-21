import json
import os
import sys
import time
import datetime
import urllib.parse
import traceback
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterable

# Third-party imports - these may show as unresolved in some IDEs
# but they are required dependencies for the script
try:
    import google.generativeai as Client
    from github import Github, GithubException
    import requests
    import fnmatch
    import jwt
    from unidiff import Hunk, PatchedFile, PatchSet
except ImportError as e:
    print(f"Error importing required dependencies: {e}")
    print("Please install required packages: pip install PyGithub google-generativeai PyJWT requests unidiff")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GitHubAuthenticator:
    """
    Centralized class for handling GitHub authentication.
    Supports both GitHub App authentication and personal access token (GITHUB_TOKEN).
    """

    def __init__(self):
        self.app_id = "1281729"  # Hardcoded GitHub App ID for zen-ai-qa
        self.installation_id = os.environ.get("ZEN_APP_INSTALLATION_ID")
        self.token = None
        self.client = None

    def get_private_key(self):
        """
        Retrieve the private key using the most secure method available.
        Prioritizes file-based storage over environment variables.
        """
        # Check if we have a file path to the key
        if 'ZEN_APP_PRIVATE_KEY_PATH' in os.environ:
            try:
                with open(os.environ['ZEN_APP_PRIVATE_KEY_PATH'], 'r') as key_file:
                    return key_file.read()
            except Exception as e:
                logger.warning(f"Could not read private key from file: {e}")

        # Fall back to environment variable
        if 'ZEN_APP_PRIVATE_KEY' in os.environ:
            # Log that we're using the less secure method
            logger.info("Using private key from environment variable (less secure)")
            private_key = os.environ.get("ZEN_APP_PRIVATE_KEY")
            # Replace newline placeholders if the key was stored with them
            return private_key.replace('\\n', '\n')

        return None

    def generate_jwt_token(self):
        """
        Generate a JWT token for GitHub App authentication.

        Returns:
            str: JWT token for GitHub App authentication or None if failed
        """
        try:
            private_key = self.get_private_key()
            if not private_key:
                logger.error("No private key available for GitHub App authentication")
                return None

            # Create JWT payload
            now = int(time.time())
            payload = {
                'iat': now,                # Issued at time
                'exp': now + (10 * 60),    # JWT expiration time (10 minutes)
                'iss': self.app_id         # GitHub App's identifier
            }

            # Generate JWT token
            jwt_token = jwt.encode(payload, private_key, algorithm='RS256')

            # jwt.encode might return bytes in some versions of PyJWT
            if isinstance(jwt_token, bytes):
                jwt_token = jwt_token.decode('utf-8')

            logger.info(f"Successfully generated JWT token for GitHub App ID: {self.app_id}")
            return jwt_token
        except Exception as e:
            logger.error(f"Error generating JWT token: {e}")
            traceback.print_exc()
            return None

    def get_installation_access_token(self, jwt_token):
        """
        Get an installation access token for the GitHub App.

        Args:
            jwt_token (str): JWT token for GitHub App authentication

        Returns:
            str: Installation access token or None if failed
        """
        try:
            if not self.installation_id:
                logger.error("ZEN_APP_INSTALLATION_ID environment variable is required for GitHub App authentication")
                return None

            headers = {
                'Authorization': f'Bearer {jwt_token}',
                'Accept': 'application/vnd.github+json',
                'X-GitHub-Api-Version': '2022-11-28'
            }

            url = f'https://api.github.com/app/installations/{self.installation_id}/access_tokens'
            response = requests.post(url, headers=headers)

            if response.status_code != 201:
                logger.error(f"Error getting installation access token. Status code: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return None

            access_token = response.json().get('token')
            if not access_token:
                logger.error("No access token found in the response")
                return None

            logger.info(f"Successfully obtained installation access token for installation ID: {self.installation_id}")
            return access_token
        except Exception as e:
            logger.error(f"Error getting installation access token: {e}")
            traceback.print_exc()
            return None

    def authenticate(self):
        """
        Authenticate using either GitHub App credentials or GITHUB_TOKEN.

        Returns:
            tuple: (Github client, token) or (None, None) if authentication failed

        Raises:
            ValueError: If no valid authentication credentials are available
        """
        # Check if we're in test mode
        if os.environ.get("GEMINI_TEST_MODE") == "1":
            logger.info("Test mode: Skipping GitHub authentication")
            return None, None

        # Try GitHub App authentication first if credentials are available
        private_key = self.get_private_key()
        if self.app_id and self.installation_id and private_key:
            logger.info("GitHub App authentication credentials found. Using GitHub App authentication.")

            # Try to generate JWT token
            try:
                jwt_token = self.generate_jwt_token()
                if not jwt_token:
                    logger.error("Failed to generate JWT token for GitHub App authentication")
                    logger.info("Falling back to GITHUB_TOKEN due to JWT token generation failure")
                else:
                    # Try to get installation token
                    try:
                        installation_token = self.get_installation_access_token(jwt_token)
                        if not installation_token:
                            logger.error("Failed to get installation access token for GitHub App authentication")
                            logger.info("Falling back to GITHUB_TOKEN due to installation token retrieval failure")
                        else:
                            logger.info("Using GitHub App installation token for authentication")
                            self.token = installation_token
                            self.client = Github(installation_token)
                            return self.client, self.token
                    except Exception as e:
                        logger.error(f"Error getting installation access token: {e}")
                        logger.info("Falling back to GITHUB_TOKEN due to installation token error")
            except Exception as e:
                logger.error(f"Error during JWT token generation: {e}")
                logger.info("Falling back to GITHUB_TOKEN due to JWT generation error")
        else:
            # Log specific reason for not using GitHub App authentication
            if not self.app_id:
                logger.info("GitHub App ID not available. Falling back to GITHUB_TOKEN.")
            elif not self.installation_id:
                logger.info("GitHub App installation ID not available. Falling back to GITHUB_TOKEN.")
            elif not private_key:
                logger.info("GitHub App private key not available. Falling back to GITHUB_TOKEN.")
            else:
                logger.info("GitHub App authentication not available for unknown reason. Falling back to GITHUB_TOKEN.")

        # Fall back to GITHUB_TOKEN
        github_token = os.environ.get("GITHUB_TOKEN")
        if not github_token:
            logger.error("GITHUB_TOKEN environment variable is required when GitHub App authentication is not available")
            raise ValueError("No valid GitHub authentication credentials found")

        self.token = github_token
        self.client = Github(github_token)
        return self.client, self.token

# Initialize Gemini client
class GeminiKeyManager:
    """
    Class to manage Gemini API keys and handle rotation when rate limiting occurs.
    Supports multiple alternative keys (GEMINI_ALT_1 through GEMINI_ALT_4).
    """
    def __init__(self):
        # Initialize primary key
        self.primary_key = os.environ.get("GEMINI_API_KEY")
        if not self.primary_key:
            logger.error("GEMINI_API_KEY environment variable is required")
            raise ValueError("GEMINI_API_KEY environment variable is required")

        # Initialize alternative keys
        self.alt_keys = {}
        for i in range(1, 5):  # GEMINI_ALT_1 through GEMINI_ALT_4
            key_name = f"GEMINI_ALT_{i}"
            key_value = os.environ.get(key_name)
            if key_value:
                self.alt_keys[key_name] = key_value
                logger.info(f"Alternative key {key_name} is available for rotation")

        # Set current key to primary
        self.current_key = self.primary_key
        self.current_key_name = "GEMINI_API_KEY"

        # Track rate limited keys and rotation order
        self.rate_limited_keys = set()
        self.encountered_rate_limiting = False
        self.all_keys_rate_limited = False
        self.used_fallback_key = False
        self.rotation_order = ["GEMINI_API_KEY"] + list(self.alt_keys.keys())

        # Log initialization status
        logger.info(f"Initialized GeminiKeyManager with primary key and {len(self.alt_keys)} alternative keys")
        if not self.alt_keys:
            logger.warning("No alternative keys (GEMINI_ALT_1 through GEMINI_ALT_4) found. API key rotation will not be available.")

    def get_current_key(self):
        """Get the currently active API key."""
        return self.current_key

    def get_current_key_name(self):
        """Get the name of the currently active API key."""
        return self.current_key_name

    def get_key_by_name(self, key_name):
        """Get an API key by its name."""
        if key_name == "GEMINI_API_KEY":
            return self.primary_key
        return self.alt_keys.get(key_name)

    def rotate_key(self):
        """
        Rotate to the next available API key in sequence.
        Returns True if rotation was successful, False if no more keys are available.
        """
        # Mark current key as rate limited
        self.rate_limited_keys.add(self.current_key_name)

        # Find the next available key in rotation order
        current_index = self.rotation_order.index(self.current_key_name)
        tried_count = 0

        # Try each key in sequence until we find one that's not rate limited
        while tried_count < len(self.rotation_order):
            # Move to next key in rotation order
            next_index = (current_index + 1) % len(self.rotation_order)
            next_key_name = self.rotation_order[next_index]

            # Skip if this key is already rate limited
            if next_key_name in self.rate_limited_keys:
                current_index = next_index
                tried_count += 1
                continue

            # Get the actual key value
            next_key = self.get_key_by_name(next_key_name)
            if not next_key:
                # This key doesn't exist or is empty, mark as rate limited and continue
                self.rate_limited_keys.add(next_key_name)
                current_index = next_index
                tried_count += 1
                continue

            # Found a valid key, update current key
            logger.info(f"Rotating from {self.current_key_name} to {next_key_name} due to rate limiting")
            self.current_key = next_key
            self.current_key_name = next_key_name
            self.used_fallback_key = True  # Mark that we successfully used a fallback key
            return True

            # Note: Code below is unreachable due to return statement above
            # Keeping for clarity of algorithm
            # current_index = next_index
            # tried_count += 1

        # If we get here, we've tried all keys and they're all rate limited
        logger.warning("All API keys are rate limited. Resetting to primary key.")
        self.current_key = self.primary_key
        self.current_key_name = "GEMINI_API_KEY"
        self.all_keys_rate_limited = True  # Mark that all keys were rate limited
        self.rate_limited_keys.clear()  # Clear the set to try again
        return False

    def is_rate_limit_error(self, error):
        """
        Check if an error is a rate limit error.
        """
        error_str = str(error).lower()
        is_rate_limit = (
            "429" in error_str or
            "quota" in error_str or
            "rate limit" in error_str or
            "resourceexhausted" in error_str
        )

        if is_rate_limit:
            # Mark that we've encountered rate limiting
            self.encountered_rate_limiting = True

            # Note: all_keys_rate_limited is set in rotate_key() when rotation fails
            # used_fallback_key is set in rotate_key() when rotation succeeds

        return is_rate_limit

# Global instance of the key manager
gemini_key_manager = None

def initialize_gemini_client():
    """
    Initialize the Gemini API client.

    Returns:
        module: Configured Gemini client module

    Raises:
        ValueError: If GEMINI_API_KEY is not available
    """
    global gemini_key_manager

    if os.environ.get("GEMINI_TEST_MODE") == "1":
        logger.info("Test mode: Skipping Gemini client initialization")
        return None

    # Initialize the key manager
    gemini_key_manager = GeminiKeyManager()

    # Configure the client with the initial key
    Client.configure(api_key=gemini_key_manager.get_current_key())
    return Client

# Initialize clients
try:
    # Create authenticator instance
    authenticator = GitHubAuthenticator()

    # Authenticate and get GitHub client
    gh, github_token = authenticator.authenticate()

    # Initialize Gemini client
    gemini_client_module = initialize_gemini_client()

    logger.info("Successfully initialized GitHub and Gemini clients")
except ValueError as e:
    logger.error(f"Initialization error: {str(e)}")
    sys.exit(1)
except Exception as e:
    logger.error(f"Unexpected error during initialization: {str(e)}")
    traceback.print_exc()
    sys.exit(1)


class PRDetails:
    def __init__(self, owner: str, repo_name_str: str, pull_number: int, title: str, description: str, repo_obj=None, pr_obj=None, event_type: str = None):
        self.owner = owner
        self.repo_name = repo_name_str
        self.pull_number = pull_number
        self.title = title
        self.description = description
        self.repo_obj = repo_obj
        self.pr_obj = pr_obj
        self.event_type = event_type

    def get_full_repo_name(self):
        return f"{self.owner}/{self.repo_name}"


def get_pr_details() -> PRDetails:
    github_event_path = os.environ.get("GITHUB_EVENT_PATH")
    if not github_event_path:
        print("Error: GITHUB_EVENT_PATH environment variable not set.")
        sys.exit(1)

    with open(github_event_path, "r", encoding="utf-8") as f:
        event_data = json.load(f)

    event_name = os.environ.get("GITHUB_EVENT_NAME")
    pr_event_type = None

    if event_name == "issue_comment":
        if "issue" in event_data and "pull_request" in event_data["issue"]:
            pull_number = event_data["issue"]["number"]
            repo_full_name = event_data["repository"]["full_name"]
            pr_event_type = "comment"
        else:
            print("Error: issue_comment event not on a pull request.")
            sys.exit(1)
    elif event_name == "pull_request":
        pull_number = event_data["pull_request"]["number"]
        repo_full_name = event_data["repository"]["full_name"]
        pr_event_type = event_data.get("action")
        print(f"Pull request event action: {pr_event_type}")
    else:
        print(f"Error: Unsupported GITHUB_EVENT_NAME: {event_name}")
        sys.exit(1)

    owner, repo_name_str = repo_full_name.split("/")

    try:
        repo_obj = gh.get_repo(repo_full_name)
        pr_obj = repo_obj.get_pull(pull_number)
    except GithubException as e:
        print(f"Error accessing GitHub repository or PR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while fetching PR details: {e}")
        sys.exit(1)

    return PRDetails(owner, repo_name_str, pull_number, pr_obj.title, pr_obj.body or "", repo_obj, pr_obj, pr_event_type)


def get_diff(pr_details: PRDetails, comparison_sha: Optional[str] = None) -> str:
    """
    Get the diff for a PR, with multiple fallback strategies.

    Args:
        pr_details: PRDetails object containing PR information
        comparison_sha: Optional SHA to compare against HEAD

    Returns:
        str: The diff text or empty string if all methods fail
    """
    repo = pr_details.repo_obj
    pr = pr_details.pr_obj
    head_sha = pr.head.sha

    # Strategy 1: Use repo.compare if comparison_sha is provided
    if comparison_sha:
        logger.info(f"Getting diff comparing HEAD ({head_sha}) against specified SHA ({comparison_sha})")
        try:
            comparison_obj = repo.compare(comparison_sha, head_sha)
            diff_parts = []
            for file_diff in comparison_obj.files:
                if file_diff.patch:
                    # Construct a valid diff header format for unidiff
                    source_file_path_for_header = file_diff.previous_filename if file_diff.status == 'renamed' else file_diff.filename
                    target_file_path_for_header = file_diff.filename

                    diff_header = f"diff --git a/{source_file_path_for_header} b/{target_file_path_for_header}\n"
                    if file_diff.status == 'added':
                        diff_header += f"new file mode {getattr(file_diff, 'mode', '100644')}\n"
                        diff_header += f"index 0000000..{file_diff.sha[:7]}\n"
                    elif file_diff.status == 'deleted':
                        diff_header += f"deleted file mode {getattr(file_diff, 'mode', '100644')}\n"
                        diff_header += f"index {file_diff.sha[:7]}..0000000\n"
                    elif file_diff.status == 'renamed':
                        diff_header += f"similarity index {getattr(file_diff, 'similarity_index', '100')}%\n"
                        diff_header += f"rename from {source_file_path_for_header}\n" # already set as prev_filename
                        diff_header += f"rename to {target_file_path_for_header}\n"   # already set as filename
                        if hasattr(file_diff, 'sha'): # If it's a rename with modifications
                             diff_header += f"index {getattr(file_diff, 'previous_sha', '0000000')[:7]}..{file_diff.sha[:7]}\n"
                    elif file_diff.status == 'modified':
                         # For modified files, the index line shows old SHA..new SHA
                         # PyGithub's file_diff.sha is the new SHA. We need the old one if available,
                         # or rely on the patch content itself to have it.
                         # For simplicity, we'll rely on the patch content for modified index line.
                         pass

                    patch_content = file_diff.patch

                    # Ensure --- and +++ lines are present, this is critical for unidiff
                    # The patch from GitHub API usually has these, but repo.compare() might be different.
                    lines = patch_content.splitlines()
                    final_patch_lines = []

                    # Simplification: Assume file_diff.patch from repo.compare is the core hunk data
                    # and we need to wrap it correctly for unidiff.
                    final_patch_lines.append(f"--- a/{source_file_path_for_header}")
                    final_patch_lines.append(f"+++ b/{target_file_path_for_header}")
                    final_patch_lines.extend(lines) # Add the actual patch lines (hunks)

                    diff_parts.append(diff_header + "\n".join(final_patch_lines))

            if diff_parts:
                diff_text = "\n".join(diff_parts) # Each element in diff_parts is a full diff for one file
                logger.info(f"Retrieved diff (length: {len(diff_text)}) using repo.compare('{comparison_sha}', '{head_sha}')")
                return diff_text
            else:
                logger.info(f"No changes found comparing {comparison_sha} to {head_sha}")
                return ""
        except GithubException as e:
            logger.warning(f"Error getting comparison diff (compare {comparison_sha} vs {head_sha}): {e}. Falling back.")
        except Exception as e:
            logger.error(f"Unexpected error during repo.compare: {e}. Falling back.")
            traceback.print_exc()

    # Strategy 2: Use pr.get_diff()
    logger.info(f"Falling back to pr.get_diff() for PR #{pr_details.pull_number}")
    try:
        diff_text = pr.get_diff() # This is usually well-formatted for unidiff
        if diff_text:
            logger.info(f"Retrieved diff (length: {len(diff_text)}) using pr.get_diff()")
            return diff_text
        else:
            logger.warning("pr.get_diff() returned no content.")
            return ""
    except GithubException as e:
        logger.warning(f"Error getting diff using pr.get_diff(): {e}. Falling back further.")
    except Exception as e:
        logger.error(f"Unexpected error during pr.get_diff(): {e}. Falling back further.")

    # Strategy 3: Use direct API request with proper authentication
    logger.info(f"Falling back to direct API request for PR diff for PR #{pr_details.pull_number}")
    api_url = f"https://api.github.com/repos/{pr_details.get_full_repo_name()}/pulls/{pr_details.pull_number}"

    # Use the global authenticator instance for authentication
    headers = {
        'Accept': 'application/vnd.github.v3.diff'
    }

    # Get a fresh token from the authenticator
    try:
        # Create a new authenticator instance for this request
        request_auth = GitHubAuthenticator()
        _, token = request_auth.authenticate()

        if token:
            headers['Authorization'] = f'token {token}'
        else:
            # Last resort: try to use GITHUB_TOKEN directly
            github_token = os.environ.get("GITHUB_TOKEN")
            if github_token:
                logger.warning("Using GITHUB_TOKEN directly for API request as authenticator failed")
                headers['Authorization'] = f'token {github_token}'
            else:
                logger.error("No authentication token available for API request")
                return ""
    except Exception as auth_error:
        logger.error(f"Authentication error for direct API request: {auth_error}")
        # Last resort: try to use GITHUB_TOKEN directly
        github_token = os.environ.get("GITHUB_TOKEN")
        if github_token:
            logger.warning("Using GITHUB_TOKEN directly for API request after authentication error")
            headers['Authorization'] = f'token {github_token}'
        else:
            logger.error("No authentication token available for API request")
            return ""

    # Make the API request
    try:
        response = requests.get(api_url, headers=headers, timeout=30)
        response.raise_for_status()
        diff_text = response.text
        logger.info(f"Retrieved diff (length: {len(diff_text)}) via direct API call.")
        return diff_text
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to get diff via direct API call: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during direct API call for diff: {e}")

    logger.error("All methods to retrieve diff failed.")
    return ""


def get_hunk_representation(hunk: Hunk) -> str:
    return str(hunk)


def get_file_content(file_path: str) -> str:
    full_file_content = ""
    code_extensions = [
        ".py", ".js", ".jsx", ".ts", ".tsx", ".html", ".css", ".scss", ".java",
        ".c", ".cpp", ".h", ".hpp", ".go", ".rs", ".php", ".rb", ".sh", ".bash",
        ".json", ".yml", ".yaml", ".toml", ".md"
    ]
    is_code_file = any(file_path.endswith(ext) for ext in code_extensions)

    if not is_code_file:
        print(f"Skipping full file context for non-code or binary-like file: {file_path}")
        return ""

    try:
        p_file_path = Path(file_path)
        if p_file_path.exists() and p_file_path.is_file():
            file_stat = p_file_path.stat()
            max_initial_read_bytes = 300000

            if file_stat.st_size > max_initial_read_bytes:
                print(f"File {file_path} is very large ({file_stat.st_size} bytes). Reading a truncated version for context.")
                with open(p_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    start_content = f.read(max_initial_read_bytes // 2)
                full_file_content = start_content + "\n\n... [content truncated due to very large size] ...\n\n"
            else:
                 with open(p_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    full_file_content = f.read()

            max_char_len_for_context = 150000
            if len(full_file_content) > max_char_len_for_context:
                print(f"File content for {file_path} still too long after initial read ({len(full_file_content)} chars), further truncating for Gemini context.")
                half_len = max_char_len_for_context // 2
                full_file_content = full_file_content[:half_len] + \
                                    "\n\n... [content context truncated for brevity] ...\n\n" + \
                                    full_file_content[-half_len:]

            print(f"Read file content for {file_path} (length: {len(full_file_content)} chars after potential truncation).")
        else:
            print(f"File {file_path} does not exist locally or is not a file. Cannot provide full context.")
    except Exception as e:
        print(f"Error reading full file content for {file_path}: {e}")
        traceback.print_exc()
    return full_file_content


def load_previous_review_data(filepath_str: str = "reviews/gemini-pr-review.json") -> Dict[str, Any]:
    """Load previous review data from JSON file if it exists."""
    filepath = Path(filepath_str)
    if not filepath.exists():
        print(f"Previous review file {filepath_str} not found. No previous context will be provided.")
        return {}

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            print(f"Successfully loaded previous review data from {filepath_str}")
            return data
    except Exception as e:
        print(f"Error loading previous review data from {filepath_str}: {e}")
        return {}


def get_previous_file_comments(review_data: Dict[str, Any], file_path: str) -> List[Dict[str, Any]]:
    """Extract previous comments for a specific file from the review data."""
    if not review_data or "review_comments" not in review_data:
        return []

    file_comments = []
    for comment in review_data.get("review_comments", []):
        if comment.get("file_path") == file_path:
            file_comments.append(comment)

    print(f"Found {len(file_comments)} previous comments for file {file_path}")
    return file_comments


def create_batch_prompt(patched_file: PatchedFile, pr_details: PRDetails) -> str:
    full_file_content_for_context = get_file_content(patched_file.path)

    # Load previous review data
    previous_review_data = load_previous_review_data()
    previous_file_comments = get_previous_file_comments(previous_review_data, patched_file.path)

    combined_hunks_text = ""
    for i, hunk in enumerate(patched_file):
        hunk_text = get_hunk_representation(hunk)
        if not hunk_text.strip():
            continue

        separator = ("-" * 20) + f" Hunk {i+1} (0-indexed: {i}) " + ("-" * 20) + "\n"
        combined_hunks_text += ("\n\n" if i > 0 else "") + separator + hunk_text

    instructions = """Your task is reviewing pull requests. You will provide structured output in JSON format.

REVIEW GUIDELINES:
- Focus on logic flaws, inconsistencies, and bugs that would affect how the application runs. These include:
  * Incorrect error handling or recovery strategies
  * Race conditions or state management issues
  * Inconsistent behavior between related components
  * Performance bottlenecks in critical paths
  * Security vulnerabilities
- Also consider code structure and maintainability issues:
  * Violations of SOLID principles or other key design patterns
  * Redundant code that could be refactored
  * Unclear logic that could lead to future bugs
- Lower priority issues include:
  * Naming conventions (only if they cause genuine confusion)
  * Documentation suggestions (only for complex logic)

IMPORTANT GUIDELINES:
- Make comments actionable with specific suggestions for improvement. Include pseudocode examples when possible.
- DO NOT comment on minor style issues or suggest adding comments to the code itself.
- Carefully analyze the full file context (if provided) and PR context before making suggestions.
- Only suggest changes relevant to the diff. Do not comment on unrelated code unless directly impacted by the changes.
- Be concise and clear in your feedback.
- If no issues are found, return an empty reviews array.

HANDLING PREVIOUS REVIEW COMMENTS:
- If my previous review comments are provided, check if they've been addressed in the current code.
- If you see "[ADDRESSED]" in a previous comment, verify the fix is correct and DO NOT repeat the issue.
- If a previous issue has been partially fixed, acknowledge the improvement and suggest any remaining changes.
- If a previous issue hasn't been addressed at all, you may note it again but focus on new issues.
- If a fix for a previous issue introduces new problems, highlight those specifically.

RESPONSE FORMAT:
Your response must be a valid JSON object with the following structure:
{
  "reviews": [
    {
      "hunkIndex": 0,  // 0-based index of the hunk in the diff (matches the 'Hunk X (0-indexed: Y)' header)
      "lineNumber": 1, // 1-based line number within the hunk content (first line after the '@@ ... @@' header)
      "reviewComment": "Your review comment in GitHub Markdown format",
      "confidence": "High" // "High", "Medium", or "Low" based on your certainty and the potential impact
    },
    // Additional review items as needed
  ]
}

IMPORTANT NOTES ABOUT LINE NUMBERS:
- The hunkIndex must be a valid 0-based index within the range of hunks in the diff
- The lineNumber must be a valid 1-based line number within the content of the specified hunk
- If you're unsure about the exact line number, choose the first line of the relevant code block
- Do not specify line numbers outside the range of the hunk content

The response will be automatically structured according to the schema provided in the API configuration.
"""

    pr_context = f"\nPull Request Title: {pr_details.title}\nPull Request Description:\n---\n{pr_details.description or 'No description provided.'}\n---\n"

    # Add previous review context if available
    previous_review_context = ""
    if previous_file_comments:
        previous_review_context = "\n## My Previous Review Comments for this file:\n"
        for i, comment in enumerate(previous_file_comments):
            comment_text = comment.get('comment_text_md', 'N/A')
            # Check if the comment has been marked as addressed
            is_addressed = "[ADDRESSED]" in comment_text
            status_marker = "✅ ADDRESSED" if is_addressed else "⏳ PENDING"

            previous_review_context += f"### Comment {i+1}: {status_marker}\n"
            previous_review_context += f"- **File**: {comment.get('file_path')}\n"
            previous_review_context += f"- **Category**: {comment.get('detected_category_heuristic', 'N/A')}\n"
            previous_review_context += f"- **Severity**: {comment.get('detected_severity_heuristic', 'N/A')}\n"
            previous_review_context += f"- **Content**: {comment_text}\n\n"

            # If the comment has been addressed, try to extract the resolution note
            if is_addressed:
                resolution_start = comment_text.find("[ADDRESSED]")
                resolution_text = comment_text[resolution_start:]
                if "**Resolution**:" in resolution_text:
                    resolution_note = resolution_text.split("**Resolution**:", 1)[1].strip()
                    previous_review_context += f"- **Resolution Note**: {resolution_note}\n\n"

    file_context_header = ""
    file_content_block = ""
    if full_file_content_for_context:
        file_context_header = "\nFull content of the file for better context (it may be truncated if too large):\n"
        file_ext = Path(patched_file.path).suffix[1:]
        file_content_block = f"```{file_ext or 'text'}\n{full_file_content_for_context}\n```\n"

    diff_to_review_header = f"\nReview the following code diffs for the file \"{patched_file.path}\" ({len(list(patched_file))} hunks):\n"
    diff_block = f"```diff\n{combined_hunks_text}\n```"

    return instructions + pr_context + previous_review_context + file_context_header + file_content_block + diff_to_review_header + diff_block


LAST_GEMINI_REQUEST_TIME = 0
GEMINI_RPM_LIMIT = 45
GEMINI_REQUEST_INTERVAL_SECONDS = 60.0 / GEMINI_RPM_LIMIT

def enforce_gemini_rate_limits():
    global LAST_GEMINI_REQUEST_TIME
    current_time = time.time()
    time_since_last = current_time - LAST_GEMINI_REQUEST_TIME
    if time_since_last < GEMINI_REQUEST_INTERVAL_SECONDS:
        wait_time = GEMINI_REQUEST_INTERVAL_SECONDS - time_since_last
        print(f"Gemini Rate Limiter: Waiting {wait_time:.2f} seconds.")
        time.sleep(wait_time)
    LAST_GEMINI_REQUEST_TIME = time.time()


def process_structured_output(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Process the structured output from the Gemini API."""
    # Validate the structure
    if not isinstance(data, dict) or "reviews" not in data or not isinstance(data["reviews"], list):
        print(f"Error: Structured output has invalid structure. Expected {{'reviews': [...]}}. Got: {type(data)}")
        return []

    # Process the reviews
    valid_reviews = []
    for i, review_item in enumerate(data["reviews"]):
        # Validate the review item
        if not isinstance(review_item, dict):
            print(f"Error: Review item {i} is not a dict: {review_item}")
            continue

        required_keys = ["hunkIndex", "lineNumber", "reviewComment", "confidence"]
        if not all(k in review_item for k in required_keys):
            print(f"Error: Review item {i} missing one or more required keys ({', '.join(required_keys)}): {review_item}")
            continue

        # Ensure types are correct
        try:
            # Convert to integers if needed
            if not isinstance(review_item["hunkIndex"], int):
                review_item["hunkIndex"] = int(review_item["hunkIndex"])
            if not isinstance(review_item["lineNumber"], int):
                review_item["lineNumber"] = int(review_item["lineNumber"])
        except (ValueError, TypeError) as e:
            print(f"Error: Review item {i} hunkIndex or lineNumber not convertible to int: {review_item}, error: {e}")
            continue

        # Validate confidence
        if review_item["confidence"] not in ["High", "Medium", "Low"]:
            print(f"Warning: Review item {i} has invalid confidence '{review_item.get('confidence')}'. Defaulting to Low.")
            review_item["confidence"] = "Low"

        valid_reviews.append(review_item)

    print(f"Successfully processed {len(valid_reviews)} valid review items from structured output.")
    return valid_reviews


def improved_calculate_github_position(file_patch: PatchedFile, hunk_idx: int, line_num_in_hunk: int) -> Optional[int]:
    """
    Improved function to calculate GitHub position for a comment.

    This function handles line number calculation more robustly by:
    1. Getting the actual hunk object by index
    2. Validating the line number is within the hunk's content range
    3. Calculating the position based on the hunk's position in the diff
    """
    try:
        # Get all hunks in the file
        hunks_in_file = list(file_patch)

        # Validate hunk index
        if not (0 <= hunk_idx < len(hunks_in_file)):
            print(f"Warning: Invalid hunk index {hunk_idx} for file {file_patch.path} (has {len(hunks_in_file)} hunks)")
            return None

        # Get the target hunk
        target_hunk = hunks_in_file[hunk_idx]

        # Get the number of lines in the hunk
        hunk_lines = list(target_hunk)
        num_lines_in_hunk = len(hunk_lines)

        # Validate line number
        if not (1 <= line_num_in_hunk <= num_lines_in_hunk):
            print(f"Warning: Line number {line_num_in_hunk} is outside the range of hunk content (1-{num_lines_in_hunk})")
            # Return a special value to indicate invalid position but don't skip the comment
            return {
                "invalidPosition": True,
                "file": file_patch.path
            }

        # Calculate position based on hunk position and line number
        position = 0

        # Add positions for all hunks before the target hunk
        for i in range(hunk_idx):
            position += 1  # For the hunk header
            position += len(list(hunks_in_file[i]))

        # Add position for the target hunk header
        position += 1

        # Add position for the line within the target hunk
        position += line_num_in_hunk - 1

        return position

    except Exception as e:
        print(f"Error calculating GitHub position: {e}")
        traceback.print_exc()
        return None


def get_ai_response_with_structured_output(prompt: str, model_name: str, max_retries: int = 3) -> List[Dict[str, Any]]:
    """
    Get AI response with improved structured output handling.

    This function uses Gemini's structured output feature with proper error handling
    and fallback mechanisms. It also handles rate limiting by rotating API keys.
    """
    global gemini_key_manager

    if not gemini_client_module:
        print("Error: Gemini client module not initialized. Cannot make API call.")
        return []

    if not gemini_key_manager:
        print("Error: Gemini key manager not initialized. Cannot make API call.")
        return []

    # Define the schema for review items
    review_item_schema = {
        "type": "object",
        "properties": {
            "hunkIndex": {"type": "integer", "description": "0-based index of the hunk in the diff"},
            "lineNumber": {"type": "integer", "description": "1-based line number within the hunk content"},
            "reviewComment": {"type": "string", "description": "The review comment text in GitHub Markdown format"},
            "confidence": {"type": "string", "enum": ["High", "Medium", "Low"], "description": "Confidence level of the review comment"}
        },
        "required": ["hunkIndex", "lineNumber", "reviewComment", "confidence"]
    }

    # Define the overall response schema
    response_schema = {
        "type": "object",
        "properties": {
            "reviews": {
                "type": "array",
                "items": review_item_schema,
                "description": "Array of review comments for the PR"
            }
        },
        "required": ["reviews"]
    }

    # Log the prompt length
    print(f"Full prompt (length {len(prompt)}). Start:\n{prompt[:1000]}...\n...End:\n{prompt[-1000:]}")

    for attempt in range(1, max_retries + 1):
        try:
            # Create the model with structured output configuration and current API key
            # We recreate the model on each attempt to ensure we're using the current API key
            Client.configure(api_key=gemini_key_manager.get_current_key())

            gemini_model = gemini_client_module.GenerativeModel(
                model_name,
                generation_config={
                    "max_output_tokens": 8192,
                    "temperature": 0.4,
                    "top_p": 0.95,
                    "top_k": 40,
                    "response_mime_type": "application/json",  # Enable structured output
                    "response_schema": response_schema  # Define the expected response structure
                },
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                ]
            )

            enforce_gemini_rate_limits()
            # Only show first 5 chars of key followed by *** for security
            key_prefix = gemini_key_manager.get_current_key()[:5] if gemini_key_manager.get_current_key() else "None"
            print(f"Attempt {attempt}/{max_retries} - Sending prompt to Gemini model {model_name} with structured output using key: {key_prefix}***")

            # Generate content with the prompt
            response = gemini_model.generate_content(prompt)

            if not response.parts:
                print(f"Warning: AI response (attempt {attempt}) was empty or blocked.")
                if attempt < max_retries:
                    time.sleep((2 ** attempt) * 2)
                    continue
                return []

            # Check if we have a structured output response
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    part = candidate.content.parts[0]
                    if hasattr(part, 'function_call') and part.function_call and hasattr(part.function_call, 'parsed'):
                        print("Received structured output response with function_call.parsed attribute.")
                        data = part.function_call.parsed
                        # Process the structured output
                        return process_structured_output(data)

            # Check if we have a parsed response (structured output)
            if hasattr(response, 'parsed'):
                print("Received structured output response with parsed attribute.")
                data = response.parsed
                # Process the structured output
                return process_structured_output(data)

            # Fallback to text parsing if structured output is not available
            print("Structured output not available. Falling back to text parsing.")
            response_text = response.text.strip()

            # Clean up the response text if it's wrapped in markdown code blocks
            if response_text.startswith("```json"):
                response_text = response_text[len("```json"):]
            if response_text.endswith("```"):
                response_text = response_text[:-len("```")]
            response_text = response_text.strip()

            # Parse the JSON
            data = json.loads(response_text)

            # Process the parsed JSON
            return process_structured_output(data)

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from AI response (attempt {attempt}): {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
            return []
        except Exception as e:
            print(f"Error during Gemini API call (attempt {attempt}): {type(e).__name__} - {e}")

            # Check if this is a rate limit error
            if gemini_key_manager.is_rate_limit_error(e):
                print(f"Detected rate limit error: {e}")

                # Log available keys status (without exposing full keys)
                available_keys = ["GEMINI_API_KEY"] + [f"GEMINI_ALT_{i}" for i in range(1, 5)]
                key_status = []
                for key_name in available_keys:
                    key_value = gemini_key_manager.get_key_by_name(key_name)
                    status = "SET" if key_value else "NOT SET"
                    key_status.append(f"{key_name}: {status}")

                print(f"API key status - {', '.join(key_status)}")
                print(f"Currently using: {gemini_key_manager.get_current_key_name()}")

                # Try to rotate to the next available key
                if gemini_key_manager.rotate_key():
                    # Only show first 5 chars of key followed by *** for security
                    key_prefix = gemini_key_manager.get_current_key()[:5] if gemini_key_manager.get_current_key() else "None"
                    print(f"Rotated to alternative API key {gemini_key_manager.get_current_key_name()}: {key_prefix}***")
                    # Don't increment attempt counter when we rotate keys
                    continue
                else:
                    print("All API keys are rate limited or unavailable")

            # For non-rate-limit errors or if key rotation failed, continue with normal retry logic
            if attempt < max_retries:
                time.sleep(5 * attempt)
                continue
            return []

    return []


def get_ai_response_with_retry(prompt: str, max_retries: int = 3) -> List[Dict[str, Any]]:
    """
    Get AI response with structured output and retry mechanism.

    This function has been updated to use Gemini's structured output feature as described in:
    https://ai.google.dev/gemini-api/docs/structured-output

    The structured output approach constrains the model to generate JSON that matches our schema,
    which should improve reliability and reduce parsing errors.
    """
    model_name = os.environ.get('GEMINI_MODEL', 'gemini-1.5-flash-latest')

    if not gemini_client_module:
        print("Error: Gemini client module not initialized. Cannot make API call.")
        return []

    # Use the improved structured output handling function
    print("Using improved structured output handling function")
    enforce_gemini_rate_limits()
    return get_ai_response_with_structured_output(prompt, model_name, max_retries)


def analyze_code(files_to_review: Iterable[PatchedFile], pr_details: PRDetails) -> List[Dict[str, Any]]:
    files_list = list(files_to_review)
    print(f"Starting code analysis for {len(files_list)} files.")
    all_comments_for_pr = []

    for patched_file in files_list:
        if not patched_file.path or patched_file.path == "/dev/null":
            print(f"Skipping file with invalid path: {patched_file.path}")
            continue

        hunks_in_file = list(patched_file)
        if not hunks_in_file:
            print(f"No hunks in file {patched_file.path}, skipping.")
            continue

        print(f"\nProcessing file: {patched_file.path} with {len(hunks_in_file)} hunks.")

        batch_prompt = create_batch_prompt(patched_file, pr_details)
        ai_reviews_for_file = get_ai_response_with_retry(batch_prompt)

        if ai_reviews_for_file:
            print(f"Received {len(ai_reviews_for_file)} review suggestions from AI for file {patched_file.path}.")
            file_comments = process_batch_ai_reviews(patched_file, ai_reviews_for_file)
            if file_comments:
                all_comments_for_pr.extend(file_comments)
        else:
            print(f"No review suggestions from AI for file {patched_file.path}.")

    print(f"\nFinished analysis. Total comments generated for PR: {len(all_comments_for_pr)}")
    return all_comments_for_pr


def get_hunk_header_str(hunk: Hunk) -> str:
    # A Hunk's string representation starts with its header: "@@ -old_start,old_len +new_start,new_len @@"
    # Or constructs it if not directly available.
    # For logging, it's useful.
    return f"@@ -{hunk.source_start},{hunk.source_length} +{hunk.target_start},{hunk.target_length} @@"


# Function removed as it's redundant - process_batch_ai_reviews will call improved_calculate_github_position directly


def process_batch_ai_reviews(patched_file: PatchedFile, ai_reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    comments_for_github = []
    hunks_in_file = list(patched_file)

    for review_detail in ai_reviews:
        try:
            hunk_idx_from_ai = review_detail["hunkIndex"]
            line_num_in_hunk_content = review_detail["lineNumber"]
            comment_text = review_detail["reviewComment"]
            confidence = review_detail["confidence"]

            if not (0 <= hunk_idx_from_ai < len(hunks_in_file)):
                print(f"Warning: AI returned out-of-bounds hunkIndex {hunk_idx_from_ai} for file {patched_file.path} "
                      f"(has {len(hunks_in_file)} hunks). Skipping comment.")
                continue

            # Call improved_calculate_github_position directly with the hunk index
            github_pos_result = improved_calculate_github_position(patched_file, hunk_idx_from_ai, line_num_in_hunk_content)

            if github_pos_result is None:
                print(f"Warning: Could not calculate GitHub position for comment in {patched_file.path}, "
                      f"Hunk Index {hunk_idx_from_ai}, Line {line_num_in_hunk_content}. Skipping.")
                continue

            formatted_comment_body = f"**My Confidence: {confidence}**\n\n{comment_text}"

            # Check if we have an invalid position result (dictionary with invalidPosition flag)
            if isinstance(github_pos_result, dict) and github_pos_result.get("invalidPosition"):
                # For invalid positions, we'll add a special prefix to the comment
                formatted_comment_body = (
                    "**Note: I couldn't precisely position this comment in the diff, but I think it's important feedback:**\n\n"
                    f"**My Confidence: {confidence}**\n\n{comment_text}"
                )

                # Add the comment to the list of comments to post at the file level (position=1)
                gh_comment = {
                    "body": formatted_comment_body,
                    "path": patched_file.path,
                    "position": 1,  # File-level comment
                    "confidence_raw": confidence,
                    "invalidPosition": True  # Add flag for tracking
                }
            else:
                # Normal case with valid position
                gh_comment = {
                    "body": formatted_comment_body,
                    "path": patched_file.path,
                    "position": github_pos_result,
                    "confidence_raw": confidence
                }

            comments_for_github.append(gh_comment)

        except KeyError as e:
            print(f"Error processing AI review item due to missing key {e}: {review_detail}")
        except Exception as e:
            print(f"Unexpected error processing AI review item {review_detail}: {e}")
            traceback.print_exc()

    return comments_for_github


def save_review_results_to_json(pr_details: PRDetails, comments: List[Dict[str, Any]], filepath_str: str = "reviews/gemini-pr-review.json") -> str:
    filepath = Path(filepath_str)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Get API key info for metadata
    api_key_info = "primary"
    if gemini_key_manager:
        if gemini_key_manager.current_key_name != "GEMINI_API_KEY":
            api_key_info = f"{gemini_key_manager.current_key_name} (rotated due to rate limiting)"

        # Check if ALL keys were rate limited (for commit message)
        # Only set rate_limited to true if all keys failed
        rate_limited = gemini_key_manager.all_keys_rate_limited
    else:
        rate_limited = False

    review_data = {
        "metadata": {
            "pr_number": pr_details.pull_number,
            "repo": pr_details.get_full_repo_name(),
            "title": pr_details.title,
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "review_tool": "I'm your Gemini AI Reviewer",
            "model_used": os.environ.get('GEMINI_MODEL', 'N/A'),
            "api_key_used": api_key_info,
            "rate_limited": rate_limited
        },
        "review_comments": []
    }

    for gh_comment_dict in comments:
        structured_comment = {
            "file_path": gh_comment_dict["path"],
            "github_diff_position": gh_comment_dict["position"],
            "comment_text_md": gh_comment_dict["body"],
            "ai_confidence": gh_comment_dict.get("confidence_raw", "N/A"),
            "detected_severity_heuristic": detect_severity(gh_comment_dict["body"]),
            "detected_category_heuristic": detect_category(gh_comment_dict["body"])
        }

        # Include invalidPosition flag if present
        if gh_comment_dict.get("invalidPosition"):
            structured_comment["invalidPosition"] = True

        review_data["review_comments"].append(structured_comment)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(review_data, f, indent=2)

    print(f"Review results saved to {filepath}")
    return str(filepath)


def detect_severity(comment_text: str) -> str:
    """Heuristically detect the severity of a comment based on its content and confidence level."""
    lower_text = comment_text.lower()

    # Extract confidence level if present
    confidence = "medium"  # Default
    if "**ai confidence: high**" in lower_text:
        confidence = "high"
    elif "**ai confidence: medium**" in lower_text:
        confidence = "medium"
    elif "**ai confidence: low**" in lower_text:
        confidence = "low"

    # Check for critical severity indicators - highest priority runtime issues
    if any(word in lower_text for word in [
        "critical", "security vulnerability", "crash", "exploit", "must fix",
        "data loss", "deadlock", "race condition", "memory leak", "infinite loop"
    ]):
        return "critical"

    # Check for high severity indicators - focus on runtime issues
    if any(word in lower_text for word in [
        "bug", "error", "incorrect", "wrong", "security", "potential vulnerability",
        "flaw", "exception", "broken", "incorrect behavior", "inconsistent behavior"
    ]):
        return "high"

    # Check for medium severity indicators - focus on code quality and maintainability
    if any(word in lower_text for word in [
        "performance", "optimization", "memory", "leak", "consider fixing",
        "confusing", "unclear", "inefficient", "redundant", "duplicate"
    ]):
        # Upgrade to high if confidence is high
        if confidence == "high":
            return "high"
        return "medium"

    # Check for low severity indicators - minor improvements
    if any(word in lower_text for word in [
        "style", "formatting", "naming", "convention", "documentation",
        "comment", "clarity", "suggestion", "consider", "might", "could"
    ]):
        # Upgrade based on confidence
        if confidence == "high":
            return "medium"
        return "low"

    # Use confidence level as a fallback
    if confidence == "high":
        return "medium"
    elif confidence == "medium":
        return "low"

    # Default to low severity
    return "low"

def detect_category(comment_text: str) -> str:
    """Categorize review comments based on their content with improved focus on runtime issues."""
    lower_text = comment_text.lower()

    # Runtime behavior categories - highest priority
    if any(word in lower_text for word in [
        "security", "vulnerability", "exploit", "auth", "csrf", "xss",
        "injection", "password", "secret", "encryption", "authentication"
    ]):
        return "security"

    if any(word in lower_text for word in [
        "race condition", "deadlock", "concurrency", "thread safety",
        "synchronization", "atomic", "lock", "mutex", "semaphore"
    ]):
        return "concurrency"

    if any(word in lower_text for word in [
        "bug", "error", "incorrect", "wrong", "fix", "defect", "exception",
        "nullpointer", "undefined", "crash", "broken", "inconsistent behavior"
    ]):
        return "bug"

    if any(word in lower_text for word in [
        "memory leak", "resource leak", "file handle", "connection",
        "not closed", "not released", "not disposed"
    ]):
        return "resource-management"

    # Performance and optimization
    if any(word in lower_text for word in [
        "performance", "slow", "optimization", "efficient", "memory",
        "cpu", "latency", "resource", "bottleneck", "timeout", "delay"
    ]):
        return "performance"

    # Code quality categories
    if any(word in lower_text for word in [
        "error handling", "exception handling", "try/catch", "recovery",
        "fallback", "resilience", "robustness"
    ]):
        return "error-handling"

    if any(word in lower_text for word in [
        "refactor", "clean", "simplify", "maintainability", "design", "architecture",
        "pattern", "anti-pattern", "duplication", "redundant", "duplicate"
    ]):
        return "refactoring/design"

    if any(word in lower_text for word in [
        "state management", "state machine", "lifecycle", "initialization",
        "cleanup", "side effect"
    ]):
        return "state-management"

    # Lower priority categories
    if any(word in lower_text for word in [
        "test", "coverage", "assertion", "mocking", "unit test",
        "integration test", "e2e test"
    ]):
        return "testing"

    if any(word in lower_text for word in [
        "style", "format", "naming", "convention", "readability", "clarity",
        "understandability", "documentation", "commenting"
    ]):
        return "style/clarity"

    # Default category
    return "general"


def create_review_and_summary_comment(pr_details: PRDetails, comments_for_gh_review: List[Dict[str, Any]], review_json_path: str):
    """
    Create a review with comments and a summary comment on the PR.

    Args:
        pr_details: PRDetails object containing PR information
        comments_for_gh_review: List of comments to add to the review
        review_json_path: Path to the JSON file containing the review data
    """
    if not pr_details.pr_obj:
        logger.error("PR object not available in PRDetails. Cannot create review or comments.")
        return

    pr = pr_details.pr_obj
    num_suggestions = len(comments_for_gh_review)

    # Check if the global gh client is authenticated with the bot identity
    try:
        # First try to use the global gh client which should already be authenticated
        if gh and hasattr(gh, '_Github__requester') and hasattr(gh._Github__requester, 'auth'):
            # Check if we're using the zen-ai-qa bot identity (app_id 1281729)
            auth_header = getattr(gh._Github__requester.auth, 'token', '')
            if auth_header and os.environ.get("ZEN_APP_INSTALLATION_ID"):
                # We're already authenticated with the bot identity
                repo = gh.get_repo(pr_details.get_full_repo_name())
                pr = repo.get_pull(pr_details.pull_number)
                logger.info(f"Using globally authenticated client with bot identity for PR #{pr_details.pull_number}")
            else:
                # We're authenticated but not with the bot identity, try to re-authenticate
                logger.info("Global client not authenticated with bot identity, attempting to use bot credentials")
                # Create a new authenticator instance for this request
                review_auth = GitHubAuthenticator()
                github_client, token = review_auth.authenticate()

                if github_client and token:
                    # Create a new PR object with the authenticated client
                    repo = github_client.get_repo(pr_details.get_full_repo_name())
                    pr = repo.get_pull(pr_details.pull_number)
                    logger.info(f"Successfully authenticated with bot identity for PR #{pr_details.pull_number}")
                else:
                    logger.warning("Bot authentication failed. Using original PR object.")
        else:
            # Global client not properly initialized, fall back to original PR object
            logger.warning("Global GitHub client not properly initialized. Using original PR object.")
    except Exception as auth_error:
        logger.error(f"Error during GitHub authentication: {auth_error}")
        logger.warning("Falling back to original PR object due to error.")

    # Process and post review comments
    if num_suggestions > 0:
        valid_review_comments = []
        for c in comments_for_gh_review:
            if all(k in c for k in ["body", "path", "position"]):
                if isinstance(c["position"], int) and isinstance(c["path"], str) and isinstance(c["body"], str):
                    valid_review_comments.append({
                        "body": c["body"],
                        "path": c["path"],
                        "position": c["position"]
                    })
                else:
                    logger.warning(f"Skipping malformed comment due to type mismatch: {c}")
            else:
                logger.warning(f"Skipping malformed comment due to missing keys: {c}")

        if valid_review_comments:
            try:
                logger.info(f"Creating a PR review with {len(valid_review_comments)} suggestions.")
                pr.create_review(
                    body="I've reviewed your code and have some suggestions:",
                    event="COMMENT",
                    comments=valid_review_comments
                )
                logger.info("Successfully created PR review with suggestions.")
            except GithubException as e:
                logger.error(f"Error creating PR review: {e}. Status: {e.status}, Data: {e.data}")
                logger.warning("Falling back to posting individual issue comments for suggestions.")
                for c_item in valid_review_comments:
                    try:
                        pr.create_issue_comment(f"I found an issue in **File:** `{c_item['path']}` (at diff position {c_item['position']})\n\n{c_item['body']}")
                    except Exception as ie:
                        logger.error(f"Error posting individual suggestion as issue comment: {ie}")
            except Exception as e:
                logger.error(f"Unexpected error during PR review creation: {e}")
                traceback.print_exc()
        else:
            logger.warning("No validly structured comments to create a review with.")
    else:
        logger.info("No suggestions to create a PR review for.")

    # Prepare summary comment with links to review file
    repo_full_name = os.environ.get("GITHUB_REPOSITORY", pr_details.get_full_repo_name())
    server_url = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    branch_name = os.environ.get("GITHUB_HEAD_REF")
    if not branch_name and hasattr(pr.head, 'ref'):
        branch_name = pr.head.ref

    review_file_url_md = f"Review JSON file (`{review_json_path}` in the repository)"
    if branch_name:
        try:
            encoded_branch = urllib.parse.quote_plus(branch_name)
            review_file_url = f"{server_url}/{repo_full_name}/blob/{encoded_branch}/{review_json_path}"
            review_file_url_md = f"Full review details in [`{review_json_path}`]({review_file_url})"
            logger.info(f"Summary comment will link to: {review_file_url}")
        except Exception as url_e:
            logger.error(f"Error creating review file URL: {url_e}")
    else:
        logger.warning("Could not determine branch name for summary comment URL.")

    # Create summary body with categorized findings
    summary_body = f"✨ **I've completed my code review!** ✨\n\n"
    if num_suggestions > 0:
        # Group comments by category and severity for better organization
        comments_by_category = {}
        for c in comments_for_gh_review:
            if "body" not in c:
                continue

            # Extract category and severity
            category = detect_category(c["body"])
            severity = detect_severity(c["body"])

            if category not in comments_by_category:
                comments_by_category[category] = {"high": 0, "medium": 0, "low": 0, "critical": 0}

            if severity in comments_by_category[category]:
                comments_by_category[category][severity] += 1

        # Add summary of findings by category
        summary_body += f"- I found {num_suggestions} potential areas for discussion/improvement (see my review comments above or in the review tab).\n"
        summary_body += f"- {review_file_url_md}.\n\n"

        summary_body += "### Summary of My Findings by Category:\n"
        for category, severities in sorted(comments_by_category.items()):
            total = sum(severities.values())
            if total == 0:
                continue

            # Format severity counts
            severity_parts = []
            if severities.get("critical", 0) > 0:
                severity_parts.append(f"**{severities['critical']} critical**")
            if severities.get("high", 0) > 0:
                severity_parts.append(f"**{severities['high']} high**")
            if severities.get("medium", 0) > 0:
                severity_parts.append(f"{severities['medium']} medium")
            if severities.get("low", 0) > 0:
                severity_parts.append(f"{severities['low']} low")

            severity_text = ", ".join(severity_parts)
            summary_body += f"- **{category}**: {total} issues ({severity_text})\n"
    else:
        summary_body += "- I didn't find any specific issues in this code review pass.\n"

    # Add review information
    summary_body += f"\n### About My Review\n"
    summary_body += f"- I used model: `{os.environ.get('GEMINI_MODEL', 'N/A')}`\n"

    # Add API key information if key rotation occurred
    api_key_info = "primary"
    rate_limit_warning = ""
    fallback_key_note = ""

    if gemini_key_manager:
        if gemini_key_manager.current_key_name != "GEMINI_API_KEY":
            api_key_info = "alternative (rotated due to rate limiting)"

        # Add note about fallback key usage if applicable
        if gemini_key_manager.used_fallback_key:
            fallback_key_note = f"- **Note:** I encountered rate limiting with the primary API key, but I was able to use a fallback key successfully.\n"

        # Add warning only if ALL keys were rate limited and got no results
        if gemini_key_manager.all_keys_rate_limited and num_suggestions == 0:
            rate_limit_warning = "\n\n⚠️ **Warning: I encountered rate limiting with ALL my API keys during this review.** My review may be incomplete or missing suggestions due to API quota limitations."

    summary_body += f"- API key used: {api_key_info}\n"
    summary_body += fallback_key_note
    summary_body += f"- My review focused on: Logic flaws, runtime behavior issues, and code quality\n"
    summary_body += f"- To mark my comments as resolved: Add `[ADDRESSED]` to the comment in the JSON file, followed by `**Resolution**: your explanation`\n"

    # Add rate limit warning if applicable
    summary_body += rate_limit_warning

    # Add workflow run log link if available
    run_id = os.environ.get('GITHUB_RUN_ID')
    repo_name = os.environ.get('GITHUB_REPOSITORY')
    if run_id and repo_name:
        # Use a direct link to the run instead of trying to link to the specific job
        # This is more reliable as the job ID format in URLs is numeric and not available directly
        run_log_url = f"https://github.com/{repo_name}/actions/runs/{run_id}"
        summary_body += f"- [View workflow run log]({run_log_url})\n"

    # Post the summary comment
    try:
        # Use the PR object that was potentially updated with GitHub App authentication
        pr.create_issue_comment(summary_body)
        logger.info("Successfully created summary comment on PR.")
    except GithubException as e:
        logger.error(f"Error creating summary PR comment: {e}")
    except Exception as e:
        logger.error(f"Unexpected error creating summary PR comment: {e}")
        traceback.print_exc()


def parse_diff_to_patchset(diff_text: str) -> Optional[PatchSet]:
    if not diff_text:
        print("No diff text to parse.")
        return None
    try:
        patch_set = PatchSet(diff_text)
        print(f"Diff parsed into PatchSet with {len(list(patch_set))} patched files.")
        return patch_set
    except Exception as e:
        print(f"Error parsing diff string with unidiff: {type(e).__name__} - {e}")
        print(f"Diff text that failed (first 1000 chars): {diff_text[:1000]}")
    return None


def main():
    """
    Main function to run the AI code review process.

    This function handles the entire workflow:
    1. Get PR details
    2. Determine what to review based on event type
    3. Get the diff
    4. Parse the diff
    5. Filter files to analyze
    6. Analyze the code
    7. Save review results
    8. Create review and summary comments

    Proper exception handling is implemented throughout to ensure the script
    doesn't terminate abruptly and provides useful error messages.
    """
    logger.info("Starting AI Code Review Script...")

    # Validate that clients are available
    if not gh or not gemini_client_module:
        logger.error("GitHub or Gemini client not available. Exiting.")
        raise ValueError("GitHub or Gemini client not available")

    try:
        # Get PR details
        pr_details = get_pr_details()
        logger.info(f"Processing PR #{pr_details.pull_number} in repo {pr_details.get_full_repo_name()} (Event: {pr_details.event_type})")

        # Determine what to review based on event type
        last_run_sha_from_env = os.environ.get("LAST_RUN_SHA", "").strip()
        head_sha = pr_details.pr_obj.head.sha
        base_sha = pr_details.pr_obj.base.sha

        comparison_sha_for_diff = None
        if pr_details.event_type in ["opened", "reopened"]:
            comparison_sha_for_diff = base_sha
            logger.info(f"Event type is '{pr_details.event_type}'. Reviewing full PR against base SHA: {comparison_sha_for_diff}")
        elif pr_details.event_type == "synchronize":
            if last_run_sha_from_env and last_run_sha_from_env != head_sha:
                comparison_sha_for_diff = last_run_sha_from_env
                logger.info(f"Event type is 'synchronize'. Reviewing changes since last run SHA: {comparison_sha_for_diff}")
            else:
                comparison_sha_for_diff = base_sha
                if not last_run_sha_from_env:
                    logger.info(f"Event type is 'synchronize', but no last_run_sha found. Reviewing full PR against base SHA: {comparison_sha_for_diff}")
                elif last_run_sha_from_env == head_sha:
                    logger.info(f"Event type is 'synchronize', but last_run_sha ({last_run_sha_from_env}) is same as head_sha. No new commits for incremental review. Defaulting to full review against base SHA: {comparison_sha_for_diff}.")
        else:
            comparison_sha_for_diff = base_sha
            logger.info(f"Event type is '{pr_details.event_type}'. Defaulting to full review against base SHA: {comparison_sha_for_diff}")

        # Check if there are any changes to review
        if head_sha == comparison_sha_for_diff:
            logger.info(f"HEAD SHA ({head_sha}) is the same as comparison SHA ({comparison_sha_for_diff}). No new changes to diff.")
            save_review_results_to_json(pr_details, [], "reviews/gemini-pr-review.json")
            create_review_and_summary_comment(pr_details, [], "reviews/gemini-pr-review.json")
            logger.info("Exiting as there are no new changes to review based on SHAs.")
            return

        # Get the diff
        diff_text = get_diff(pr_details, comparison_sha_for_diff)
        if not diff_text:
            logger.warning("No diff content retrieved. Exiting review process.")
            save_review_results_to_json(pr_details, [], "reviews/gemini-pr-review.json")
            create_review_and_summary_comment(pr_details, [], "reviews/gemini-pr-review.json")
            return

        # Parse the diff
        initial_patch_set = parse_diff_to_patchset(diff_text)
        if not initial_patch_set:
            logger.error("Failed to parse diff into PatchSet. Exiting.")
            save_review_results_to_json(pr_details, [], "reviews/gemini-pr-review.json")
            raise ValueError("Failed to parse diff into PatchSet")

        # Filter files to analyze
        exclude_patterns_str = os.environ.get("INPUT_EXCLUDE", "")
        exclude_patterns = [p.strip() for p in exclude_patterns_str.split(',') if p.strip()]

        actual_files_to_process: List[PatchedFile] = []
        for patched_file_obj in initial_patch_set:
            normalized_path = patched_file_obj.path.lstrip('./')
            is_excluded = False

            if patched_file_obj.is_removed_file or (patched_file_obj.is_added_file and patched_file_obj.target_file == '/dev/null'):
                logger.info(f"Skipping removed file (or added as /dev/null): {patched_file_obj.path}")
                is_excluded = True
            elif patched_file_obj.is_binary_file:
                logger.info(f"Excluding binary file: {patched_file_obj.path}")
                is_excluded = True
            else:
                for pattern in exclude_patterns:
                    if fnmatch.fnmatch(normalized_path, pattern) or fnmatch.fnmatch(patched_file_obj.path, pattern):
                        logger.info(f"Excluding file '{patched_file_obj.path}' due to pattern '{pattern}'.")
                        is_excluded = True
                        break
            if not is_excluded:
                actual_files_to_process.append(patched_file_obj)

        num_files_to_analyze = len(actual_files_to_process)
        logger.info(f"Number of files to analyze after exclusions: {num_files_to_analyze}")

        if num_files_to_analyze == 0:
            logger.warning("No files to analyze after applying exclusion patterns.")
            save_review_results_to_json(pr_details, [], "reviews/gemini-pr-review.json")
            create_review_and_summary_comment(pr_details, [], "reviews/gemini-pr-review.json")
            return

        # Analyze the code
        comments_for_gh_review_api = analyze_code(actual_files_to_process, pr_details)

        # Save review results and create comments
        review_json_filepath = "reviews/gemini-pr-review.json"
        save_review_results_to_json(pr_details, comments_for_gh_review_api, review_json_filepath)
        create_review_and_summary_comment(pr_details, comments_for_gh_review_api, review_json_filepath)

        logger.info("AI Code Review Script finished successfully.")
    except ValueError as e:
        # Expected errors that we've explicitly raised
        logger.error(f"Error in main process: {str(e)}")
        # We don't re-raise here as we want to handle these gracefully
    except Exception as e:
        # Unexpected errors
        logger.error(f"Unexpected error in main process: {str(e)}")
        traceback.print_exc()
        # We don't re-raise here to avoid abrupt termination


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        # Let system exit exceptions propagate
        raise
    except Exception as e:
        # Catch any unhandled exceptions that weren't caught in main()
        logger.critical(f"Unhandled exception in __main__: {type(e).__name__} - {e}")
        traceback.print_exc()

        # Create an empty review file to avoid workflow failures
        try:
            # Get PR details if possible
            try:
                pr_details = get_pr_details()
                save_review_results_to_json(pr_details, [], "reviews/gemini-pr-review.json")
            except Exception:
                # If we can't get PR details, create a minimal review file
                os.makedirs("reviews", exist_ok=True)
                with open("reviews/gemini-pr-review.json", "w") as f:
                    json.dump({"metadata": {"error": str(e)}, "review_comments": []}, f)
        except Exception as file_error:
            logger.critical(f"Failed to create empty review file: {file_error}")

        # Exit with error code
        sys.exit(1)
