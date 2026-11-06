from neura.platform import NeuraRuntime, NeuraPlugin


class ExamplePlugin(NeuraPlugin):

    name = "example"
    version = "1.0.0"
    description = "Neura platform test plugin"

    def initialize(self, runtime):
        self.initialized = True

    def shutdown(self, runtime):
        self.initialized = False


def main():
    runtime = NeuraRuntime()

    plugin = ExamplePlugin()

    runtime.load_plugin(plugin)

    assert runtime.plugins.get("example") is plugin
    assert runtime.registry.plugins.has("example")
    assert plugin.initialized is True

    status = runtime.status()

    assert "example" in status["plugins"]
    assert status["registry"]["plugins"] == 1

    runtime.unload_plugin("example")

    assert runtime.plugins.get("example") is None
    assert runtime.registry.plugins.has("example") is False
    assert plugin.initialized is False

    print("=" * 60)
    print("NEURA PLATFORM PLUGIN TEST")
    print("=" * 60)
    print("Plugin:", plugin.name)
    print("Version:", plugin.version)
    print("Status after unload:", runtime.status())
    print("=" * 60)
    print("RESULT: PLATFORM PLUGIN SYSTEM PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
