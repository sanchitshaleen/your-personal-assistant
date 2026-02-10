try:
    from langgraph.checkpoint.redis import RedisSaver
    print("RedisSaver imported")
    try:
        r = RedisSaver(conn=None) # Intentionally wrong to see init signature error or we can check repr
    except Exception as e:
        print(f"Init error: {e}")
    
    import inspect
    print(f"Sig: {inspect.signature(RedisSaver)}")
except ImportError:
    print("RedisSaver not found")
