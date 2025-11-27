import requests
import json
import subprocess
import sys
import re


def get_petram_repos(local_packages, org_name="piScope", token="", verbose=True):
    #
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"  # Specify the API version
    }
    if token != "":
        headers["Authorization"] = f"token {token}"

    url = f"https://api.github.com/orgs/{org_name}/repos"
    print(url)

    response = requests.get(url, headers=headers)


    if response.status_code == 200:
        repo_data = response.json()
        repo_data = [repo for repo in repo_data if repo['name'].startswith("Petra-M")]

        for repo in repo_data:
            mname = ["petram",repo['name'][7:].split("-")[-1].lower()]
            mname = [x for x in mname if x != ""]
            print(mname)
            mname = "_".join(mname)
            repo["module"] = mname
            if mname in local_packages:
                repo["installed"] = "yes"
                repo["version"] = local_packages[mname]
            else:
                repo["installed"] = "no"
                repo["version"] = ""
                
        if verbose:
            print(f"Found {len(repo_data)} repositories in {org_name}:")
            print("-" * 20)        
            for repo in repo_data:
                print(f"- Name: {repo['name']}")
                print(f"  Module: {repo['module']}")                
                print(f"  Description: {repo.get('description', 'No description available')}")
                print(f"  Private: {repo['private']}")
                print(f"  URL: {repo['html_url']}")
                print(f"  Insalled: {repo['installed']}")
                print(f"  Version: {repo['version']}")                
                print("-" * 20)


    else:
        txt1 = [f"Failed to retrieve repositories. Status code: {response.status_code}"]
        txt1.append(str(response.text))

        assert False, "\n".join(txt1)

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
                
def get_repo_info():
    local_packages = get_local_packages()
    get_petram_repos(local_packages)



