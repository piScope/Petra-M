import json
import subprocess
import sys
import re
import urllib
import urllib.request as request
from importlib.metadata import distribution


def get_url_content(url, headers=None):
    if headers is None:
        headers = {}
    try:
        # Create a Request object with the URL and custom headers
        req = request.Request(url, headers=headers)

        # Open the URL and get the response
        with request.urlopen(req) as response:
            body = response.read().decode('utf-8')
            status = response.getcode()
            headers = dict(response.info())

            return body, status, headers

    except urllib.error.HTTPError as e:
        # print(f"HTTP Error: {e.code} - {e.reason}")
        body = e.read().decode()
        return body, e.code, None
    except urllib.error.URLError as e:
        body = f"URL Error: {e.reason}"
        return body, None, None
    except Exception as e:
        body = f"An unexpected error occurred: {e}"
        return body, None, None


def check_latest(repo):
    owner = repo["owner"]["login"]
    name = repo["name"]
    url2 = f"https://api.github.com/repos/{owner}/{name}/releases/latest"

    response, status, _header = get_url_content(url2)
    data = json.loads(response)
    if status == 200:
        return data["name"]
    return ""


def get_petram_repos(local_packages, url, token="", verbose=False):
    #
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"  # Specify the API version
    }
    if token != "":
        headers["Authorization"] = f"token {token}"

    response, status, _header = get_url_content(url, headers)

    if status == 200:
        repo_data = json.loads(response)
        repo_data = [
            repo for repo in repo_data if repo['name'].startswith("Petra-M")]

        for repo in repo_data:
            mname = ["petram", repo['name'][7:].split("-")[-1].lower()]
            mname = [x for x in mname if x != ""]
            mname = "_".join(mname)
            repo["module"] = mname
            if mname in local_packages:
                repo["installed"] = "yes"
                repo["version"] = local_packages[mname]
            else:
                repo["installed"] = "no"
                repo["version"] = ""

            repo["latest"] = check_latest(repo)

            desc = repo.get('description', 'No description available')
            if desc is None:
                desc = 'No description available'
            repo['description'] = desc

        mnames = [repo["module"] for repo in repo_data]
        for mname in local_packages:
            if mname not in mnames:
                # pakcage not found in public repos.
                url = "?"
                dist = distribution(mname)
                txts = dist.metadata.get_all('Project-URL')
                for line in txts:
                    if 'github.com' in line.lower():
                        parts = line.split(',', 1)
                        if len(parts) > 1:
                            url = parts[1].strip()
                            break
                desc = str(dist.metadata.get("Summary"))
                data = {"name": mname,
                        "module": mname,
                        "installed": "yes",
                        "version": local_packages[mname],
                        "latest": "?",
                        "html_url": url,
                        "description": desc,
                        "private": "?", }
                repo_data.append(data)

        if verbose:
            print(f"Found {len(repo_data)} repositories in {url}:")
            print("-" * 20)
            for repo in repo_data:
                print(f"- Name: {repo['name']}")
                print(f"  Module: {repo['module']}")
                print(f"  Description: {repo['description']}")
                print(f"  Latest: {repo['latest']}")
                print(f"  Private: {repo['private']}")
                print(f"  URL: {repo['html_url']}")
                print(f"  Insalled: {repo['installed']}")
                print(f"  Version: {repo['version']}")
                print("-" * 20)

    else:
        txt1 = [f"Failed to retrieve repositories. Status code: {status}"]
        if response is not None:
            txt1.append(response)

        assert False, "\n".join(txt1)

    return repo_data


def get_local_packages():
    result = subprocess.run([sys.executable, "-m", "pip", "list"],
                            capture_output=True, text=True, check=True)
    data = result.stdout.strip().split('\n')

    res = {}
    for x in data:
        if x.startswith('petram'):
            name, version = re.split(r'\s+', x)
            res[name] = version

    return res


def get_repo_info(urls=None, verbose=False):
    if urls is None:
        urls = ["https://api.github.com/orgs/piScope/repos"]

    local_packages = get_local_packages()

    repo = []
    for url in urls:
        repo_data = get_petram_repos(local_packages, url, verbose=verbose)
        repo.extend(repo_data)

    repo = [y[1] for y in sorted([(x["module"], x) for x in repo])]
    return repo


if __name__ == "__main__":
    get_repo_info(verbose=True)
