from waybackpy import WaybackMachineCDXServerAPI

url = "https://octominer.com/wp-content/uploads/"
user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36"
cdx = WaybackMachineCDXServerAPI(url, user_agent)

print("Fetching snapshots from the Wayback Machine...")
for snapshot in cdx.snapshots():
    snapshot_url = snapshot.archive_url
    print(f"Snapshot found: {snapshot_url}")
    # Check content or download if matches
