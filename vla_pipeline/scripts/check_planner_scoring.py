"""Offline scoring contracts: known identities, wrong zones, unknowns, and schemas.

No model download, renderer, or training. Run before comparing VLMs so a parser
regression cannot turn correct predictions into unknown objects or vice versa.
"""
import _bootstrap  # noqa: F401
import json
from vlapipe import config as C
from evaluate_planner import response_items


def main():
    checked = 0
    for object_set in ('color', 'number'):
        if object_set == 'number':
            C.set_number_count(10)
        C.use_object_set(object_set)
        for obj in C.OBJECTS:
            raw = json.dumps([{'object': obj['ref_en'], 'target_zone': obj['home_zone']}])
            assert response_items(raw, 'number' if object_set == 'number' else 'json') == {
                (obj['name'], obj['home_zone'])}
            checked += 1
    correct = response_items('[{"object":"cube number 7","target_zone":"left storage zone"}]', 'number')
    wrong = response_items('[{"object":"cube number 7","target_zone":"right storage zone"}]', 'number')
    assert correct != wrong and len(correct - wrong) == 1
    unknown = response_items('[{"object":"cube number 99","target_zone":"left storage zone"}]', 'number')
    assert len(unknown - correct) == 1
    assert response_items('[3,99]', 'number_read') == {3, 99}
    for raw, style in (('[3,7]', 'number'), ('["3"]', 'number_read'), ('not JSON', 'number')):
        try:
            response_items(raw, style)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError(f'Schema should fail: {raw}')
    print(f'PASS: {checked} object identities, wrong destination, unknown99 retained, strict schemas')


if __name__ == '__main__':
    main()
