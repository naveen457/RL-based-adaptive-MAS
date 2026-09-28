import requests

BASE_URL = "http://127.0.0.1:8000"

def test_user_isolation():
    user_a = "user_alpha_1"
    user_b = "user_beta_2"
    thread_a = "sha_alpha_111"
    thread_b = "sha_beta_222"

    print("1. Sending message for User A...")
    resp_a = requests.post(f"{BASE_URL}/chat", json={
        "message": "Hello from User A",
        "conversationId": thread_a,
        "user_id": user_a
    })
    assert resp_a.status_code == 200, f"User A chat failed: {resp_a.text}"

    print("2. Sending message for User B...")
    resp_b = requests.post(f"{BASE_URL}/chat", json={
        "message": "Hello from User B",
        "conversationId": thread_b,
        "user_id": user_b
    })
    assert resp_b.status_code == 200, f"User B chat failed: {resp_b.text}"

    print("3. Querying threads for User A...")
    threads_a = requests.get(f"{BASE_URL}/threads?user_id={user_a}").json()["threads"]
    thread_ids_a = [t["thread_id"] for t in threads_a]
    print(f"User A threads: {thread_ids_a}")
    assert thread_a in thread_ids_a, f"{thread_a} not found in {thread_ids_a}"
    assert thread_b not in thread_ids_a, f"Leak! {thread_b} found in User A threads!"

    print("4. Querying threads for User B...")
    threads_b = requests.get(f"{BASE_URL}/threads?user_id={user_b}").json()["threads"]
    thread_ids_b = [t["thread_id"] for t in threads_b]
    print(f"User B threads: {thread_ids_b}")
    assert thread_b in thread_ids_b, f"{thread_b} not found in {thread_ids_b}"
    assert thread_a not in thread_ids_b, f"Leak! {thread_a} found in User B threads!"

    print("5. User A attempting to access User B messages...")
    msgs_leak = requests.get(f"{BASE_URL}/threads/{thread_b}/messages?user_id={user_a}").json()["messages"]
    print(f"User A querying User B thread messages returned: {len(msgs_leak)} messages")
    assert len(msgs_leak) == 0, "User A was able to read User B messages!"

    print("\nSUCCESS: Per-user thread isolation strictly verified!")

if __name__ == "__main__":
    test_user_isolation()
