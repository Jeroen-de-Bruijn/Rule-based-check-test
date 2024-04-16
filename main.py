"""This module contains the business logic of the function.

Use the automation_context module to wrap your function in an Automate context helper
"""

# import os
import numpy as np

from pydantic import Field, SecretStr
from speckle_automate import (
    AutomateBase,
    AutomationContext,
    execute_automate_function,
)

from flatten import flatten_base

# import
#from specklepy.api.wrapper import StreamWrapper
from specklepy.api import operations
#from specklepy.serialization.base_object_serializer import BaseObjectSerializer

from specklepy.api import operations
from specklepy.api.client import SpeckleClient
#from specklepy.objects.base import Base
from specklepy.transports.server import ServerTransport

class FunctionInputs(AutomateBase):
    """These are function author defined values.

    Automate will make sure to supply them matching the types specified here.
    Please use the pydantic model schema to define your inputs:
    https://docs.pydantic.dev/latest/usage/models/
    """

    # an example how to use secret values
    # whisper_message: SecretStr = Field(title="This is a secret message")

    forbidden_speckle_type: str = Field(
        title="Forbidden speckle type",
        description="If a object has the following speckle_type, it will be marked with a warning.",
    )
    
    # delete_objects: bool = Field(
    #    default=False,
    #    title="TEST Delete objects",
    #    description="Enable this option to delete the objects.",
    #)

def get_parameter_value(speckle_object):
    """
    Given a Speckle object, retrieve the value of parameter_name.
    Args:
        speckle_object: The Speckle object containing parameters.
    Returns:
        The value of parameter_name if found, or None if not present.
    """
    
    # Test
    parameter_name = "Area"

    try:
        # Assuming speckle_object is a dictionary or has a similar structure
        param_value = speckle_object.get(parameter_name)
        return param_value
    except KeyError:
        # parameter_name not found in the object
        return None


def automate_function(
    automate_context: AutomationContext,
    function_inputs: FunctionInputs,
) -> None:
    """This is an example Speckle Automate function.

    Args:
        automate_context: A context helper object, that carries relevant information
            about the runtime context of this function.
            It gives access to the Speckle project data, that triggered this run.
            It also has convenience methods attach result data to the Speckle model.
        function_inputs: An instance object matching the defined schema.
    """
    print("test function 01")
    
    # project = https://latest.speckle.systems/projects/e79a76b289
    # model = https://latest.speckle.systems/projects/e79a76b289/models/3b4ec1bbbb/
    # version = https://latest.speckle.systems/projects/e79a76b289/models/3b4ec1bbbb%4007dcb5fae8
    # project_id = "e79a76b289"
    # version_id = "07dcb5fae8"
    # versions overview = https://latest.speckle.systems/projects/e79a76b289/models/3b4ec1bbbb/versions
    
    speckle_server_url = "https://latest.speckle.systems/"
    
    project_id = "e79a76b289"
    version_id = "07dcb5fae8"

    other_client = SpeckleClient(speckle_server_url, speckle_server_url.startswith("https"))
    other_client.authenticate_with_token(automate_context._speckle_token)
    
    other_commit = other_client.commit.get(project_id, version_id)

    if not other_commit.referencedObject:
        raise ValueError("The commit has no referencedObject, cannot receive it.")
    
    #other_server_transport: ServerTransport
    other_server_transport = ServerTransport(stream_id=project_id, client=other_client)
    other_root_object = operations.receive(other_commit.referencedObject, other_server_transport)

    print("test function 02")
    
    # The context provides a convenient way, to receive the triggering version
    version_root_object = automate_context.receive_version()
    print("test function 03")

    # Get all walls so we can check and compare their types
    find_this_speckle_type = "Objects.BuiltElements.Wall:Objects.BuiltElements.Revit.RevitWall"
    
    objects_match = [
        b
        for b in flatten_base(version_root_object)
        if b.speckle_type == find_this_speckle_type
    ]
    count = len(objects_match)
    
    other_objects_match = [
        b
        for b in flatten_base(other_root_object)
        if b.speckle_type == find_this_speckle_type
    ]
    other_count = len(other_objects_match)
    
    list_points = []
    other_list_points = []
    remove_objects = []
    other_remove_objects = []

    print("loop1")
    for count, o in enumerate(objects_match):
        mesh_list = o["displayValue"]

        if len(mesh_list) >= 1:
            mesh_first = mesh_list[0]
        else:
            remove_objects.append(count)
            # No mesh found so go to the next item in the loop
            continue
        
        vertices_list = mesh_first["vertices"]
        p1 = vertices_list[0], vertices_list[1], vertices_list[2]

        list_points.append(p1)
    
    print("loop other_objects_match")
    for count, o in enumerate(other_objects_match):
        mesh_list = o["displayValue"]
        
        if len(mesh_list) >= 1:
            mesh_first = mesh_list[0]
        else:
            other_remove_objects.append(count)
            # No mesh found so go to the next item in the loop
            continue
        
        vertices_list = mesh_first["vertices"]
        p1 = vertices_list[0], vertices_list[1], vertices_list[2]
        
        other_list_points.append(p1)
    
    print("test function 04 point lists created")

    # Remove walls without mesh
    objects_match_clean = [item for i, item in enumerate(objects_match) if i not in remove_objects]
    other_objects_match_clean = [item for i, item in enumerate(other_objects_match) if i not in other_remove_objects]
    
    # Convert lists to NumPy arrays for efficient computation
    list_A = list_points
    list_B = other_list_points
    
    # Convert lists to NumPy arrays for efficient computation
    array_A = np.array(list_A)
    array_B = np.array(list_B)

    # Initialize an array to keep track of whether points in list_B have been used
    used_indices = np.zeros(len(list_B), dtype=bool)

    # Initialize an array to store the closest points
    test_obj = {

    }
    closest_points = []
    closest_walls = []
    match_distance = []
    matching_walls = []

    # Iterate over each point in list_A
    for point_A in array_A:
        # Calculate distances from point_A to all unused points in list_B
        distances = np.linalg.norm(point_A - array_B[~used_indices], axis=1)
        
        # Find the index of the closest unused point in list_B
        closest_index = np.argmin(distances)
        
        # Mark the closest point as used
        used_indices[closest_index] = True
        
        # Append the closest point to the result
        closest_points.append(list_B[closest_index])
        closest_walls.append(other_objects_match_clean[closest_index])
        match_distance.append(distances[closest_index])

    # matches = {"distance": X, "wall1": other_object, "wall2": different_object}

    print("test function 05 closest points calculated")

    count_types_ok = 0
    count_types_fail = 0

    # Print the results
    # for i, point in enumerate(closest_points):
    #    print(f"Closest point for point {list_A[i]} is {point}")
    for i, wall2 in enumerate(closest_walls):
        wall1 = objects_match_clean[i]
        type1 = wall1["type"]
        type2 = wall2["type"]

        print(f"Wall1 type: {type1}\n",
              f"Wall2 type: {type2}\n",
              f"Distance: {match_distance[i]}")

        if type1 == type2:
            count_types_ok += 1
            automate_context.attach_info_to_objects(
                category = "Type check ok",
                object_ids = wall1.id,
                message = f"Wall1 type: {type1}\nWall2 type: {type2}\nDistance: {match_distance[i]}"
            )
        else:
            count_types_fail += 1
            automate_context.attach_warning_to_objects(
            #automate_context.attach_error_to_objects(
                category = "Type check failed",
                object_ids = wall1.id,
                message = f"Wall1 type: {type1}\nWall2 type: {type2}\nDistance: {match_distance[i]}"
            )

    print("test function code completed")

    if count_types_fail == 0 and count_types_ok == 0:
        automate_context.mark_run_failed("Automation run failed. Something went wrong. Maybe no walls were found?")
    else:
        automate_context.mark_run_success(
            "Automation completed: "
            f"{count_types_fail} objects have non matching wall types! It's advised to review them.\n"
            f"{count_types_ok} objects have matching wall types."
        )

        # set the automation context view, to the original model / version view
        # to show the offending objects
        automate_context.set_context_view()


def automate_function_without_inputs(automate_context: AutomationContext) -> None:
    """A function example without inputs.

    If your function does not need any input variables,
     besides what the automation context provides,
     the inputs argument can be omitted.
    """
    pass


# make sure to call the function with the executor
if __name__ == "__main__":
    # NOTE: always pass in the automate function by its reference, do not invoke it!

    # pass in the function reference with the inputs schema to the executor
    execute_automate_function(automate_function, FunctionInputs)

    # if the function has no arguments, the executor can handle it like so
    # execute_automate_function(automate_function_without_inputs)
