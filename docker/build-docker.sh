# docker buildx build --platform linux/amd64,linux/arm64/v8 --build-arg max_step_size=0.005 --build-arg real_time_update_rate=0 --tag lucasmrdt/sailboat-sim-lsa-gym:mss5 --push .
# docker buildx build --platform linux/amd64,linux/arm64/v8 --build-arg max_step_size=0.001 --build-arg real_time_update_rate=1000 --load --tag lucasmrdt/sailboat-sim-lsa-gym:realtime --cache-from lucasmrdt/sailboat-sim-lsa-gym:realtime --push .

# docker buildx build --platform linux/amd64,linux/arm64/v8 --build-arg max_step_size=0.001 --build-arg real_time_update_rate=1000 --tag lucasmrdt/sailboat-sim-lsa-gym:realtime --cache-from lucasmrdt/sailboat-sim-lsa-gym:realtime --push .

docker buildx build --platform linux/amd64,linux/arm64/v8 --build-arg max_step_size=0.001 --build-arg real_time_update_rate=0 --tag lucasmrdt/sailboat-sim-lsa-gym:mss1 --cache-from lucasmrdt/sailboat-sim-lsa-gym:mss1 --push .

# docker buildx build --platform linux/amd64,linux/arm64/v8 --build-arg max_step_size=0.002 --build-arg real_time_update_rate=0 --tag lucasmrdt/sailboat-sim-lsa-gym:mss2 --cache-from lucasmrdt/sailboat-sim-lsa-gym:mss2 --push .

# docker buildx build --platform linux/amd64,linux/arm64/v8 --build-arg max_step_size=0.003 --build-arg real_time_update_rate=0 --tag lucasmrdt/sailboat-sim-lsa-gym:mss3 --cache-from lucasmrdt/sailboat-sim-lsa-gym:mss3 --push .

# docker buildx build --platform linux/amd64,linux/arm64/v8 --build-arg max_step_size=0.004 --build-arg real_time_update_rate=0 --tag lucasmrdt/sailboat-sim-lsa-gym:mss4 --cache-from lucasmrdt/sailboat-sim-lsa-gym:mss4 --push .

# docker buildx build --platform linux/amd64,linux/arm64/v8 --build-arg max_step_size=0.006 --build-arg real_time_update_rate=0 --tag lucasmrdt/sailboat-sim-lsa-gym:mss6 --cache-from lucasmrdt/sailboat-sim-lsa-gym:mss6 --push .

# docker buildx build --platform linux/amd64,linux/arm64/v8 --build-arg max_step_size=0.007 --build-arg real_time_update_rate=0 --tag lucasmrdt/sailboat-sim-lsa-gym:mss7 --cache-from lucasmrdt/sailboat-sim-lsa-gym:mss7 --push .

# docker buildx build --platform linux/amd64,linux/arm64/v8 --build-arg max_step_size=0.008 --build-arg real_time_update_rate=0 --tag lucasmrdt/sailboat-sim-lsa-gym:mss8 --cache-from lucasmrdt/sailboat-sim-lsa-gym:mss8 --push .

# docker buildx build --platform linux/amd64,linux/arm64/v8 --build-arg max_step_size=0.009 --build-arg real_time_update_rate=0 --tag lucasmrdt/sailboat-sim-lsa-gym:mss9 --cache-from lucasmrdt/sailboat-sim-lsa-gym:mss9 --push .

# docker buildx build --platform linux/amd64,linux/arm64/v8 --build-arg max_step_size=0.010 --build-arg real_time_update_rate=0 --tag lucasmrdt/sailboat-sim-lsa-gym:mss10 --cache-from lucasmrdt/sailboat-sim-lsa-gym:mss10 --push .