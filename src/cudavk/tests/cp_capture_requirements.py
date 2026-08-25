#!/usr/bin/env python3
"""Summarise what a Vulkan application requires, from a GFXReconstruct capture.

Capture on a driver that works, then read the capture to find out what the
application actually asks for — rather than guessing from what some class of
hardware happens to expose.

    gfxrecon-capture-vulkan.py -o app.gfxr <application>
    gfxrecon-convert --output app.json app.gfxr
    cp_capture_requirements.py app.json

The interesting output is the image format list: it settles which compressed
texture family (BC, ETC2, ASTC) the content actually ships, and which formats
have to be renderable rather than merely samplable.
"""

import collections
import json
import sys

USAGE_BITS = [
    (0x001, "transfer_src"), (0x002, "transfer_dst"), (0x004, "sampled"),
    (0x008, "storage"), (0x010, "color"), (0x020, "depth_stencil"),
    (0x040, "transient"), (0x080, "input_attachment"),
]


def short(value, prefix):
    """VK_FORMAT_R8G8B8A8_UNORM -> R8G8B8A8_UNORM."""
    text = str(value)
    return text[len(prefix):] if text.startswith(prefix) else text


def usage_name(value):
    try:
        bits = int(str(value), 0)
    except (TypeError, ValueError):
        return str(value)
    names = [n for bit, n in USAGE_BITS if bits & bit]
    return "+".join(names) if names else "0x%x" % bits


def walk(records):
    images = collections.Counter()
    samplers = collections.Counter()
    pipelines = collections.Counter()
    extensions = set()
    stages = collections.Counter()
    probed = collections.Counter()

    for rec in records:
        fn = rec.get("function")
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        args = fn.get("args") or {}

        if name == "vkCreateImage":
            ci = args.get("pCreateInfo") or {}
            key = "%-24s %-4s mips=%-3s layers=%-4s %-6s %s" % (
                short(ci.get("format"), "VK_FORMAT_"),
                short(ci.get("imageType"), "VK_IMAGE_TYPE_"),
                ci.get("mipLevels"), ci.get("arrayLayers"),
                short(ci.get("samples"), "VK_SAMPLE_COUNT_"),
                usage_name(ci.get("usage")))
            images[key] += 1

        elif name == "vkCreateSampler":
            ci = args.get("pCreateInfo") or {}
            samplers["min=%s mag=%s mip=%s wrap=%s aniso=%s compare=%s" % (
                short(ci.get("minFilter"), "VK_FILTER_"),
                short(ci.get("magFilter"), "VK_FILTER_"),
                short(ci.get("mipmapMode"), "VK_SAMPLER_MIPMAP_MODE_"),
                short(ci.get("addressModeU"), "VK_SAMPLER_ADDRESS_MODE_"),
                ci.get("anisotropyEnable"), ci.get("compareEnable"))] += 1

        elif name == "vkCreateGraphicsPipelines":
            infos = args.get("pCreateInfos") or []
            if isinstance(infos, dict):
                infos = [infos]
            for ci in infos:
                if not isinstance(ci, dict):
                    continue
                ia = ci.get("pInputAssemblyState") or {}
                ds = ci.get("pDepthStencilState") or {}
                cb = ci.get("pColorBlendState") or {}
                ms = ci.get("pMultisampleState") or {}
                atts = cb.get("pAttachments") or []
                if isinstance(atts, dict):
                    atts = [atts]
                blend = any(a.get("blendEnable") for a in atts
                            if isinstance(a, dict))
                for st in (ci.get("pStages") or []):
                    if isinstance(st, dict):
                        stages[short(st.get("stage"), "VK_SHADER_STAGE_")] += 1
                pipelines["topology=%-16s targets=%-2s blend=%-5s depth=%-5s "
                          "stencil=%-5s samples=%s" % (
                              short(ia.get("topology"), "VK_PRIMITIVE_TOPOLOGY_"),
                              len(atts), bool(blend),
                              bool(ds.get("depthTestEnable")),
                              bool(ds.get("stencilTestEnable")),
                              short(ms.get("rasterizationSamples"),
                                    "VK_SAMPLE_COUNT_"))] += 1

        elif name in ("vkCreateInstance", "vkCreateDevice"):
            ci = args.get("pCreateInfo") or {}
            for e in (ci.get("ppEnabledExtensionNames") or []):
                extensions.add(str(e))

        elif name in ("vkGetPhysicalDeviceFormatProperties",
                      "vkGetPhysicalDeviceFormatProperties2"):
            probed[short(args.get("format"), "VK_FORMAT_")] += 1

    return images, samplers, pipelines, extensions, stages, probed


def show(title, counter, limit=None):
    if not counter:
        return
    print("\n== %s ==" % title)
    items = counter.most_common() if isinstance(counter, collections.Counter) \
        else [(v, None) for v in sorted(counter)]
    for key, count in items[:limit]:
        print("   %s%s" % (key, "  x%d" % count if count else ""))
    if limit and len(items) > limit:
        print("   ... %d more" % (len(items) - limit))


def main():
    with open(sys.argv[1]) as f:
        records = json.load(f)

    images, samplers, pipelines, extensions, stages, probed = walk(records)

    print("requirements from %s" % sys.argv[1])
    show("images created", images)
    show("samplers", samplers)
    show("graphics pipeline state", pipelines)
    show("shader stages used", stages)
    show("extensions requested", extensions)
    show("formats probed", probed, limit=40)

    # The question that actually drives driver work.
    families = {"BC": 0, "ETC2": 0, "ASTC": 0}
    for key, count in images.items():
        for fam in families:
            if key.lstrip().startswith(fam):
                families[fam] += count
    print("\n== compressed texture families in use ==")
    if any(families.values()):
        for fam, count in families.items():
            if count:
                print("   %-5s %d images" % (fam, count))
    else:
        print("   none — all textures are uncompressed")


if __name__ == "__main__":
    main()
